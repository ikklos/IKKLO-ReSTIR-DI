import Utils.Math.HashUtils;
import Utils.Math.MathHelpers;
import Utils.Math.ShadingFrame;
import Utils.Geometry.IntersectionHelpers;
import Utils.Color.ColorHelpers;
import Utils.Sampling.TinyUniformSampleGenerator;
import Rendering.Materials.IsotropicGGX;
import Rendering.Materials.Fresnel;

import Scene.RaytracingInline;
import Scene.Scene;
import RenderPasses.ReSTIR_DI.LoadShadingData;

import Reservoir;
import RandomHelper;
import Utils.Math.Ray;
#include "rtxdi/RtxdiMath.hlsli"
#define RTXDI_POLYMORPHIC_LIGHT_SAMPLING
#include "PolymorphicLight.slang"
#include "ResamplingFunctions.slangh"
#include "RTDIApplicationBridge.slangh"
#define PI 3.14159265358979323846
#define RTDI_ReservoirMaxM 48

struct RTDI
{
    // Runtime parameters.
    uint gLocalRIS_M;
    uint gInfiniteRIS_M;
    uint gEnvRIS_M;
    uint gBrdfRIS_M;
    uint gSpatialReuseSampleCount;
    uint gPresampledTileCount;
    uint gPresampledTileSize;
    float gTemporalSamplingRadius;
    float gSpatialSamplingRadius;
    uint2 gFrameDim;
    uint gFrameIndex;
    /*TODO:RandomSeed should be generate in RH_InitRandomState*/
    uint gRandomSeed;
    float normalThreshold;
    float depthThreshold;

    uint useEmissiveTextures;
    uint firstLocalAnalyticLight;
    uint firstInfiniteAnalyticLight;
    uint envLightIndex;
    uint totalLightCount;

    uint currentSurfaceBufferIndex;
    uint prevSurfaceBufferIndex;
    uint gNeighborOffsetMask;
    float3 prevCameraU;
    float3 prevCameraV;
    float3 prevCameraW;
    float2 prevCameraJitter;

    // Internal algorithm resources.
    RWStructuredBuffer<Reservoir> gReservoir;
    StructuredBuffer<Reservoir> gPrevReservoir;

    StructuredBuffer<PackedPolymorphicLight> gLightInfo;
    RWStructuredBuffer<PackedSurfaceData> surfaceData;

    RWStructuredBuffer<uint2> gPresampledLightIndex;
    RWStructuredBuffer<uint2> gPresampledEnvData;
    Buffer<float2> gNeighborOffsetBuffer;
    Texture2D<PackedHitInfo> vbuffer;
    Texture2D<float2> mvec;
    Texture2D localLightPdfTexture;
    Texture2D envLightLuminanceTexture;
    Texture2D envLightPdfTexture;
    RWTexture2D<float4> color;

    float3 computeRayDirection(uint2 pixel, bool previousFrame)
    {
        const float2 jitter = previousFrame
            ? float2(-prevCameraJitter.x, prevCameraJitter.y)
            : float2(-gScene.camera.data.jitterX, gScene.camera.data.jitterY);

        const float2 p = (pixel + float2(0.5f, 0.5f)) / gFrameDim + jitter;
        const float2 ndc = float2(2.f, -2.f) * p + float2(-1.f, 1.f);

        const float3 cameraU = previousFrame ? prevCameraU : gScene.camera.data.cameraU;
        const float3 cameraV = previousFrame ? prevCameraV : gScene.camera.data.cameraV;
        const float3 cameraW = previousFrame ? prevCameraW : gScene.camera.data.cameraW;

        return normalize(ndc.x * cameraU + ndc.y * cameraV + cameraW);
    }

    void discardReservoirSample(inout Reservoir r)
    {
        r.lightID = -1;
        r.sampleUV = float2(0.f, 0.f);
        r.wsum = 0.f;
    }

    uint getSurfaceIndex(uint2 pixel, uint bufferIndex)
    {
        return pixel.x + pixel.y * gFrameDim.x + bufferIndex * (gFrameDim.x * gFrameDim.y);
    }

    bool buildSurfaceData(uint2 pixel, out RTAB_Surface sd)
    {
        sd = RTAB_Surface();
        ShadingData hit;
        if (!loadShadingData(pixel, gFrameDim, gScene.camera, vbuffer, hit))
            return false;

        const uint hints = getMaterialInstanceHints();
        let lod = ExplicitLodTextureSampler(0.f);
        let mi = gScene.materials.getMaterialInstance(hit, lod, hints);
        let props = mi.getProperties(hit);

        sd.position = hit.computeRayOrigin();
        sd.viewDepth = length(hit.posW - gScene.camera.data.posW);
        sd.normal = props.guideNormal;
        sd.viewDir = normalize(hit.V);
        sd.diffuse = saturate(luminance(props.diffuseReflectionAlbedo));
        sd.specular = saturate(luminance(props.specularReflectance));
        sd.roughness = saturate(props.roughness);
        sd.diffuseProb = sd.diffuse / max(sd.diffuse + sd.specular, 1e-6f);

        return true;
    }

    bool loadSurface(uint2 pixel, uint bufferIndex, bool withViewDir, out RTAB_Surface sd)
    {
        sd = RTAB_Surface();
        if (any(pixel >= gFrameDim)) return false;

        uint idx = getSurfaceIndex(pixel, bufferIndex);
        float3 viewDir = float3(0.f, 0.f, 1.f);
        if (withViewDir)
        {
            bool previousFrame = (bufferIndex == prevSurfaceBufferIndex);
            viewDir = -computeRayDirection(pixel, previousFrame);
        }

        sd = RTAB_Surface(surfaceData[idx], viewDir);
        return sd.valid;
    }
    /** calculate target pdf at this surf
    */
    float calcTargetPdf(RTAB_Surface surf, float3 wi, float3 Li)
    {
        if (!surf.valid) return 0.f;

        float3 N = normalize(surf.normal);
        float3 V = normalize(surf.viewDir);
        float3 L = normalize(wi);
        float3 H = normalize(V + L);

        float NdotV = saturate(dot(N, V));
        float NdotL = saturate(dot(N, L));
        float NdotH = saturate(dot(N, H));
        float LdotH = saturate(dot(L, H));
        if (NdotL <= 0.f || NdotV <= 0.f) return 0.f;

        float roughness = surf.roughness;
        float specular = surf.specular;
        float D = evalNdfGGX(roughness * roughness, NdotH);
        float G = evalMaskingSmithGGXSeparable(roughness, NdotL, NdotV);
        float F = specular < 1e-8f ? 0.f : (evalFresnelSchlick(specular, 1.f, LdotH) / max(specular, 1e-6f));

        float diffuseBRDF = NdotL * (1.f / PI);
        float specularBRDF = (D * G * F) / max(4.f * NdotV, 1e-6f);
        float fullBRDF = max(0.f, lerp(specularBRDF, diffuseBRDF, surf.diffuseProb));
        return fullBRDF * max(Li.x, 0.f);
    }

    float3 worldToTangent(RTAB_Surface surface, float3 w)
    {
        float3 bitangent = perp_stark(surface.normal);
        float3 tangent = cross(bitangent, surface.normal);
        return float3(dot(bitangent, w), dot(tangent, w), dot(surface.normal, w));
    }

    float3 tangentToWorld(RTAB_Surface surface, float3 h)
    {
        float3 bitangent = perp_stark(surface.normal);
        float3 tangent = cross(bitangent, surface.normal);
        return bitangent * h.x + tangent * h.y + surface.normal * h.z;
    }

    bool sampleSurfaceBrdf(RTAB_Surface surface, inout RH_RandomSamplerState rng, out float3 dir)
    {
        dir = {};

        float3 rand;
        rand.x = RH_GetNextRandom(rng);
        rand.y = RH_GetNextRandom(rng);
        rand.z = RH_GetNextRandom(rng);
        if (rand.x < surface.diffuseProb)
        {
            float pdf;
            float3 h = sample_cosine_hemisphere_concentric(rand.yz, pdf);
            dir = tangentToWorld(surface, h);
        }
        else
        {
            float pdf;
            float ggxAlpha = surface.roughness * surface.roughness;
            float3 h = sampleGGX_NDF(ggxAlpha, rand.yz, pdf);
            dir = reflect(-surface.viewDir, tangentToWorld(surface, h));
        }

        return dot(surface.normal, dir) > 0.f;
    }

    float surfaceBrdfPdf(RTAB_Surface surface, float3 dir)
    {
        float cosTheta = saturate(dot(surface.normal, dir));
        float diffusePdf = cosTheta / M_PI;

        float3 h = worldToTangent(surface, normalize(dir + surface.viewDir));
        float ggxAlpha = surface.roughness * surface.roughness;
        float specularPdf = evalPdfGGX_NDF(ggxAlpha, h.z) / (4.f * saturate(dot(h, worldToTangent(surface, surface.viewDir))));
        return cosTheta > 0.f ? lerp(specularPdf, diffusePdf, surface.diffuseProb) : 0.f;
    }

    bool traceRayForLocalLight(float3 origin, float3 direction, float tMin, float tMax, out uint o_lightIndex, out float2 o_randXY)
    {
        o_lightIndex = 0xffffffffu;
        o_randXY = 0;

        float hitT;
        SceneRayQuery<0> rayQuery;
        Ray ray = Ray(origin, direction, tMin, tMax);
        HitInfo hit = rayQuery.traceRay(ray, hitT, RAY_FLAG_CULL_NON_OPAQUE);
        if (firstInfiniteAnalyticLight > 0u && hit.isValid())
        {
            const bool isTriangleHit = hit.getType() == HitType::Triangle;
            if (isTriangleHit)
            {
                const TriangleHit triangleHit = hit.getTriangleHit();
                uint triangleIndex = gScene.lightCollection.getTriangleIndex(triangleHit.instanceID, triangleHit.primitiveIndex);
                if (triangleIndex != LightCollection::kInvalidIndex)
                {
                    uint activeLightIndex = gScene.lightCollection.getActiveTriangleIndex(triangleIndex);
                    if (activeLightIndex != LightCollection::kInvalidIndex)
                    {
                        o_randXY = triangleHit.barycentrics;
                        o_lightIndex = activeLightIndex;
                    }
                }
            }
        }

        return hit.isValid();
    }

    bool samplePolymorphicLight(RTAB_Surface surf, int lightID, float2 sampleUV, bool scalarLi, out PolymorphicLightSample ls)
    {
        ls = {};
        if (lightID < 0 || uint(lightID) >= totalLightCount) return false;
        PolymorphicLight light = PolymorphicLight(gLightInfo[uint(lightID)]);
        ls = light.calcSample(surf.position, sampleUV, scalarLi);
        return ls.valid;
    }

    bool samplePolymorphicLight(RTAB_Surface surf, int lightID, float2 sampleUV, out PolymorphicLightSample ls)
    {
        return samplePolymorphicLight(surf, lightID, sampleUV, false, ls);
    }

        float sampleFromPdfMipTreeWithProb(Texture2D tex, inout RH_RandomSamplerState randomState, out uint2 coord)
    {
        uint2 dim0;
        uint mipCount;
        tex.GetDimensions(0, dim0.x, dim0.y, mipCount);

        coord = uint2(0, 0);
        if (mipCount <= 1) return 1.f;

        float prob = 1.f;
        for (int mip = int(mipCount) - 1; mip > 0; --mip)
        {
            int2 c = int2(coord * 2);
            float w00 = loadPdfSafe(tex, mip - 1, c + int2(0, 0));
            float w10 = loadPdfSafe(tex, mip - 1, c + int2(1, 0));
            float w01 = loadPdfSafe(tex, mip - 1, c + int2(0, 1));
            float w11 = loadPdfSafe(tex, mip - 1, c + int2(1, 1));

            float sum = w00 + w10 + w01 + w11;
            uint2 off = uint2(0, 0);

            if (sum <= 0.f)
            {
                uint q = min((uint)(RH_GetNextRandom(randomState) * 4.f), 3u);
                off = uint2(q & 1u, (q >> 1u) & 1u);
                prob *= 0.25f;
            }
            else
            {
                float u = RH_GetNextRandom(randomState) * sum;
                if (u < w00) off = uint2(0, 0);
                else if (u < w00 + w10) off = uint2(1, 0);
                else if (u < w00 + w10 + w01) off = uint2(0, 1);
                else off = uint2(1, 1);

                float chosen = (off.x == 0 && off.y == 0) ? w00 :
                               (off.x == 1 && off.y == 0) ? w10 :
                               (off.x == 0 && off.y == 1) ? w01 : w11;
                prob *= chosen / max(sum, 1e-6f);
            }

            coord = coord * 2 + off;
        }

        uint2 dim;
        tex.GetDimensions(0, dim.x, dim.y, mipCount);
        coord = min(coord, dim - 1);
        return prob;
    }

    float pdfTexelFromMipTree(Texture2D tex, uint2 texel)
    {
        uint2 dim0;
        uint mipCount;
        tex.GetDimensions(0, dim0.x, dim0.y, mipCount);
        if (any(texel >= dim0)) return 0.f;
        if (mipCount <= 1) return 1.f;

        float prob = 1.f;
        uint2 coord = uint2(0, 0);

        for (int mip = int(mipCount) - 1; mip > 0; --mip)
        {
            int2 c = int2(coord * 2);
            float w00 = loadPdfSafe(tex, mip - 1, c + int2(0, 0));
            float w10 = loadPdfSafe(tex, mip - 1, c + int2(1, 0));
            float w01 = loadPdfSafe(tex, mip - 1, c + int2(0, 1));
            float w11 = loadPdfSafe(tex, mip - 1, c + int2(1, 1));
            float sum = w00 + w10 + w01 + w11;

            uint2 off = uint2(texel.x >= coord.x * 2 + 1 ? 1u : 0u, texel.y >= coord.y * 2 + 1 ? 1u : 0u);

            if (sum <= 0.f)
            {
                prob *= 0.25f;
            }
            else
            {
                float chosen = (off.x == 0 && off.y == 0) ? w00 :
                               (off.x == 1 && off.y == 0) ? w10 :
                               (off.x == 0 && off.y == 1) ? w01 : w11;
                prob *= chosen / max(sum, 1e-6f);
            }

            coord = coord * 2 + off;
        }

        return prob;
    }

    void getCategoryAvailability(out bool localAvail, out bool infAvail, out bool envAvail)
    {
        localAvail = false;
        infAvail = (envLightIndex > firstInfiniteAnalyticLight);
        envAvail = false;

        uint localCount = firstInfiniteAnalyticLight;
        if (localCount > 0)
        {
            uint mipCount;
            uint2 dim;
            localLightPdfTexture.GetDimensions(0, dim.x, dim.y, mipCount);
            localAvail = max(localLightPdfTexture.Load(int3(0, 0, int(mipCount - 1))).x, 0.f) > 0.f;
        }

        if (envLightIndex < totalLightCount)
        {
            uint mipCount;
            uint2 dim;
            envLightPdfTexture.GetDimensions(0, dim.x, dim.y, mipCount);
            envAvail = max(envLightPdfTexture.Load(int3(0, 0, int(mipCount - 1))).x, 0.f) > 0.f;
        }
    }

    float calcLocalLightSourcePdf(int lightID)
    {
        if (lightID < 0 || uint(lightID) >= firstInfiniteAnalyticLight) return 0.f;

        uint2 dim;
        uint mipCount;
        localLightPdfTexture.GetDimensions(0, dim.x, dim.y, mipCount);
        if (dim.x == 0 || dim.y == 0) return 0.f;

        uint2 texel = RTXDI_LinearIndexToZCurve(uint(lightID));
        if (any(texel >= dim)) return 0.f;
        float texelValue = max(localLightPdfTexture.Load(int3(texel, 0)).x, 0.f);

        int lastMip = max(0, int(floor(log2(max(dim.x, dim.y)))) - 1);
        float avg0 = max(localLightPdfTexture.Load(int3(0, 0, lastMip)).x, 0.f);
        float avg1 = max(localLightPdfTexture.Load(int3(1, 0, lastMip)).x, 0.f);
        float averageValue = 0.5f * (avg0 + avg1);
        float sum = averageValue * float(dim.x * dim.y);
        return (sum > 0.f) ? (texelValue / sum) : 0.f;
    }

    float calcEnvironmentMapSamplingPdf(float3 L)
    {
        uint2 dim;
        uint mipCount;
        envLightPdfTexture.GetDimensions(0, dim.x, dim.y, mipCount);
        if (dim.x == 0 || dim.y == 0) return 0.f;

        float2 uv = gScene.envMap.worldToUv(L);
        uint2 texel = min((uint2)(uv * float2(dim)), dim - 1);
        float texelValue = max(envLightPdfTexture.Load(int3(texel, 0)).x, 0.f);

        int lastMip = max(0, int(floor(log2(max(dim.x, dim.y)))) - 1);
        float avg0 = max(envLightPdfTexture.Load(int3(0, 0, lastMip)).x, 0.f);
        float avg1 = max(envLightPdfTexture.Load(int3(1, 0, lastMip)).x, 0.f);
        float averageValue = 0.5f * (avg0 + avg1);
        if (averageValue <= 0.f) return 0.f;

        return texelValue / averageValue;
    }

    float calcCategorySelectionPdf(int lightID)
    {
        bool localAvail, infAvail, envAvail;
        getCategoryAvailability(localAvail, infAvail, envAvail);

        uint categoryCount = (localAvail ? 1u : 0u) + (infAvail ? 1u : 0u) + (envAvail ? 1u : 0u);
        if (categoryCount == 0u) return 0.f;

        float pCategory = 1.f / float(categoryCount);
        if (lightID < 0) return 0.f;
        if (uint(lightID) < firstInfiniteAnalyticLight) return localAvail ? pCategory : 0.f;
        if (uint(lightID) < envLightIndex) return infAvail ? pCategory : 0.f;
        if (uint(lightID) == envLightIndex) return envAvail ? pCategory : 0.f;
        return 0.f;
    }


    uint sampleFallbackLight(inout RH_RandomSamplerState randomState, uint lightCount)
    {
        uint p = min((uint)(RH_GetNextRandom(randomState) * lightCount), lightCount - 1);
        return p;
    }

    float loadPdfSafe(Texture2D tex, uint mip, int2 p)
    {
        uint2 dim;
        uint mipCount;
        tex.GetDimensions(mip, dim.x, dim.y, mipCount);
        if (p.x < 0 || p.y < 0 || p.x >= int(dim.x) || p.y >= int(dim.y)) return 0.f;
        return max(tex.Load(int3(p, mip)).x, 0.f);
    }

    uint2 sampleFromPdfMipTree(Texture2D tex, inout RH_RandomSamplerState randomState)
    {
        uint2 coord;
        sampleFromPdfMipTreeWithProb(tex, randomState, coord);
        return coord;
    }


    bool sampleLocalLightFromPdfTexture(inout RH_RandomSamplerState randomState, out uint lightID)
    {
        lightID = 0;
        uint localCount = firstInfiniteAnalyticLight;
        if (localCount == 0) return false;

        uint2 z = sampleFromPdfMipTree(localLightPdfTexture, randomState);
        uint idx = RTXDI_ZCurveToLinearIndex(z);
        if (idx >= localCount) return false;

        lightID = idx;
        return true;
    }

    bool sampleEnvFromPdfTexture(inout RH_RandomSamplerState randomState, out float2 uv)
    {
        uv = float2(0.5f, 0.5f);
        if (envLightIndex >= totalLightCount) return false;

        uint2 dim;
        uint mipCount;
        envLightPdfTexture.GetDimensions(0, dim.x, dim.y, mipCount);
        if (dim.x == 0 || dim.y == 0) return false;

        uint2 texel = sampleFromPdfMipTree(envLightPdfTexture, randomState);
        uv = (float2(texel) + RH_GetNextRandom2D(randomState)) / float2(dim);
        return true;
    }

    uint packUv16(float2 uv)
    {
        float2 s = saturate(uv);
        return uint(s.x * 65535.f) | (uint(s.y * 65535.f) << 16);
    }

    float2 unpackUv16(uint packed)
    {
        return float2(packed & 0xffff, packed >> 16) / 65535.f;
    }

    bool sampleInfiniteAnalyticFromPmf(inout RH_RandomSamplerState randomState, out uint lightID)
    {
        lightID = 0;
        uint begin = firstInfiniteAnalyticLight;
        uint end = min(envLightIndex, totalLightCount);
        if (begin >= end) return false;

        uint count = end - begin;
        lightID = begin + min((uint)(RH_GetNextRandom(randomState) * count), count - 1);
        return true;
    }

    bool sampleLightByPdf(inout RH_RandomSamplerState randomState, out uint lightID, out float2 sampleUV)
    {
        lightID = 0;
        sampleUV = RH_GetNextRandom2D(randomState);
        if (totalLightCount == 0) return false;

        uint localCount = firstInfiniteAnalyticLight;
        uint infiniteCount = (envLightIndex > firstInfiniteAnalyticLight) ? (envLightIndex - firstInfiniteAnalyticLight) : 0u;
        uint envCount = (envLightIndex < totalLightCount) ? 1u : 0u;

        bool localAvail, infAvail, envAvail;
        getCategoryAvailability(localAvail, infAvail, envAvail);

        uint available[3];
        uint n = 0u;
        if (localCount > 0 && localAvail) available[n++] = 0u;
        if (infiniteCount > 0) available[n++] = 1u;
        if (envCount > 0 && envAvail) available[n++] = 2u;

        if (n == 0u)
        {
            lightID = sampleFallbackLight(randomState, totalLightCount);
            sampleUV = RH_GetNextRandom2D(randomState);
            return true;
        }

        uint cat = available[min((uint)(RH_GetNextRandom(randomState) * n), n - 1)];

        if (cat == 0u)
        {
            if (sampleLocalLightFromPdfTexture(randomState, lightID))
            {
                sampleUV = RH_GetNextRandom2D(randomState);
                return true;
            }
        }
        else if (cat == 1u)
        {
            if (sampleInfiniteAnalyticFromPmf(randomState, lightID))
            {
                sampleUV = RH_GetNextRandom2D(randomState);
                return true;
            }
        }
        else
        {
            float2 uv;
            if (sampleEnvFromPdfTexture(randomState, uv))
            {
                lightID = envLightIndex;
                sampleUV = uv;
                return true;
            }
        }

        lightID = sampleFallbackLight(randomState, totalLightCount);
        sampleUV = RH_GetNextRandom2D(randomState);
        return true;
    }

    Reservoir buildCategoryReservoir(
        RTAB_Surface surf,
        uint sampleCount,
        float sourceMisWeight,
        float normalizationDenominator,
        inout RH_RandomSamplerState randomState,
        uint lightCategory,
        uint tileIndex)
    {
        Reservoir r = Reservoir();
        if (sampleCount == 0u) return r;

        [loop]
        for (uint i = 0; i < sampleCount; ++i)
        {
            int lightID = -1;
            float2 uv = RH_GetNextRandom2D(randomState);
            uint totalPresampled = max(gPresampledTileCount * gPresampledTileSize, 1u);
            uint slot = (gPresampledTileSize > 0u) ? min((uint)(RH_GetNextRandom(randomState) * gPresampledTileSize), gPresampledTileSize - 1u) : 0u;
            uint entry = min(tileIndex * max(gPresampledTileSize, 1u) + slot, totalPresampled - 1u);

            if (lightCategory == 0u)
            {
                uint sampledLight = 0u;
                if (gPresampledTileCount > 0u && gPresampledTileSize > 0u)
                {
                    uint2 presampled = gPresampledLightIndex[entry];
                    sampledLight = presampled.x;
                    if (sampledLight >= firstInfiniteAnalyticLight) continue;
                }
                else if (!sampleLocalLightFromPdfTexture(randomState, sampledLight))
                {
                    continue;
                }
                lightID = int(sampledLight);
            }
            else if (lightCategory == 1u)
            {
                uint sampledLight = 0u;
                if (!sampleInfiniteAnalyticFromPmf(randomState, sampledLight)) continue;
                lightID = int(sampledLight);
            }
            else
            {
                float2 envUv = float2(0.5f, 0.5f);
                if (gPresampledTileCount > 0u && gPresampledTileSize > 0u)
                {
                    uint2 envSample = gPresampledEnvData[entry];
                    if (envSample.y != 0u) envUv = unpackUv16(envSample.x);
                    else if (!sampleEnvFromPdfTexture(randomState, envUv)) continue;
                }
                else if (!sampleEnvFromPdfTexture(randomState, envUv))
                {
                    continue;
                }
                lightID = int(envLightIndex);
                uv = envUv;
            }

            float ph = 0.f;
            float w = 0.f;
            if (lightID >= 0)
            {
                PolymorphicLightSample ls;
                if (samplePolymorphicLight(surf, lightID, uv, true, ls))
                {
                    float p = 0.f;
                    if (lightCategory == 0u)
                    {
                        if (gPresampledTileCount > 0u && gPresampledTileSize > 0u)
                        {
                            uint2 presampled = gPresampledLightIndex[entry];
                            float invSourcePdf = asfloat(presampled.y);
                            float sourcePdf = (invSourcePdf > 0.f) ? (1.f / invSourcePdf) : 0.f;
                            p = sourceMisWeight * sourcePdf;
                        }
                        else
                        {
                            p = sourceMisWeight * calcLocalLightSourcePdf(lightID);
                        }
                    }
                    else if (lightCategory == 1u)
                    {
                        uint count = envLightIndex - firstInfiniteAnalyticLight;
                        float pSelect = (count > 0u) ? (1.f / float(count)) : 0.f;
                        p = pSelect;
                    }
                    else
                    {
                        if (gPresampledTileCount > 0u && gPresampledTileSize > 0u)
                        {
                            uint2 envSample = gPresampledEnvData[entry];
                            float invSourcePdf = asfloat(envSample.y);
                            float sourcePdf = (invSourcePdf > 0.f) ? (1.f / invSourcePdf) : 0.f;
                            p = sourceMisWeight * sourcePdf;
                        }
                        else
                        {
                            p = sourceMisWeight * calcEnvironmentMapSamplingPdf(ls.direction);
                        }
                    }

                    if (p > 0.f)
                    {
                        ph = calcTargetPdf(surf, ls.direction, ls.Li);
                        w = ph / p;
                    }
                }
            }

            r.update(lightID, uv, w, ph, 1.f, randomState);
        }

        if (r.M > 0.f && r.targetPdf > 0.f)
        {
            float denom = max(normalizationDenominator, 1e-6f);
            r.wsum = r.wsum / max(r.targetPdf * denom, 1e-6f);
            r.M = 1.f;
        }
        else
        {
            r.wsum = 0.f;
            r.M = (sampleCount > 0u) ? 1.f : 0.f;
        }

        return r;
    }

    Reservoir buildBrdfReservoir(
        RTAB_Surface surf,
        uint sampleCount,
        float localMisWeight,
        float envMisWeight,
        float brdfMisWeight,
        float normalizationDenominator,
        inout RH_RandomSamplerState randomState)
    {
        Reservoir r = Reservoir();
        if (sampleCount == 0u) return r;

        [loop]
        for (uint i = 0; i < sampleCount; ++i)
        {
            float3 sampleDir;
            if (!sampleSurfaceBrdf(surf, randomState, sampleDir))
            {
                continue;
            }

            float brdfPdf = surfaceBrdfPdf(surf, sampleDir);
            if (brdfPdf <= 0.f) continue;

            uint lightIndex = 0xffffffffu;
            float2 randXY = float2(0.f, 0.f);
            bool hitAnything = traceRayForLocalLight(surf.position, sampleDir, 1e-3f, FLT_MAX, lightIndex, randXY);

            int lightID = -1;
            float2 uv = randXY;
            float lightSourcePdf = 0.f;
            bool isEnvSample = false;
            PolymorphicLightSample candidateSample = {};

            if (lightIndex != 0xffffffffu)
            {
                lightID = int(lightIndex);
                if (samplePolymorphicLight(surf, lightID, uv, true, candidateSample))
                {
                    lightSourcePdf = calcLocalLightSourcePdf(lightID);
                }
            }
            else if (!hitAnything && envLightIndex < totalLightCount)
            {
                isEnvSample = true;
                lightID = int(envLightIndex);
                uv = gScene.envMap.worldToUv(sampleDir);
                if (samplePolymorphicLight(surf, lightID, uv, true, candidateSample))
                {
                    lightSourcePdf = calcEnvironmentMapSamplingPdf(sampleDir);
                }
            }

            if (lightID < 0 || lightSourcePdf <= 0.f || !candidateSample.valid)
            {
                continue;
            }

            float targetPdf = calcTargetPdf(surf, candidateSample.direction, candidateSample.Li);
            float lightMisWeight = isEnvSample ? envMisWeight : localMisWeight;
            float blendedSourcePdf = lightMisWeight * lightSourcePdf + brdfMisWeight * brdfPdf;
            if (blendedSourcePdf <= 0.f || targetPdf <= 0.f)
            {
                continue;
            }

            float w = targetPdf / blendedSourcePdf;
            r.update(lightID, uv, w, targetPdf, 1.f, randomState);
        }

        if (r.M > 0.f && r.targetPdf > 0.f)
        {
            float denom = max(normalizationDenominator, 1e-6f);
            r.wsum = r.wsum / max(r.targetPdf * denom, 1e-6f);
            r.M = 1.f;
        }
        else
        {
            r.wsum = 0.f;
            r.M = (sampleCount > 0u) ? 1.f : 0.f;
        }

        return r;
    }

    void presampleLocalLights(uint index)
    {
        uint total = max(gPresampledTileCount * gPresampledTileSize, 1u);
        if (index >= total) return;

        uint localCount = firstInfiniteAnalyticLight;
        if (localCount == 0u)
        {
            gPresampledLightIndex[index] = uint2(0u, 0u);
            return;
        }

        RH_RandomSamplerState state = RH_InitRandomState(uint2(index,index),gFrameIndex,1);
        uint sampledLight = 0u;
        float invSourcePdf = 0.f;
        if (sampleLocalLightFromPdfTexture(state, sampledLight))
        {
            float sourcePdf = calcLocalLightSourcePdf(int(sampledLight));
            invSourcePdf = (sourcePdf > 0.f) ? (1.f / sourcePdf) : 0.f;
        }
        else
        {
            sampledLight = sampleFallbackLight(state, localCount);
            float sourcePdf = (localCount > 0u) ? (1.f / float(localCount)) : 0.f;
            invSourcePdf = (sourcePdf > 0.f) ? (1.f / sourcePdf) : 0.f;
        }
        gPresampledLightIndex[index] = uint2(sampledLight, asuint(invSourcePdf));
    }

    void presampleEnvLights(uint index)
    {
        uint total = max(gPresampledTileCount * gPresampledTileSize, 1u);
        if (index >= total) return;

        if (envLightIndex >= totalLightCount)
        {
            gPresampledEnvData[index] = uint2(0u, 0u);
            return;
        }

        uint2 dim;
        uint mipCount;
        envLightPdfTexture.GetDimensions(0, dim.x, dim.y, mipCount);
        if (dim.x == 0 || dim.y == 0)
        {
            gPresampledEnvData[index] = uint2(0u, 0u);
            return;
        }

        RH_RandomSamplerState state = RH_InitRandomState(uint2(index, index), gFrameIndex, 2);
        uint2 texel;
        float pTexel = sampleFromPdfMipTreeWithProb(envLightPdfTexture, state, texel);
        float2 uv = (float2(texel) + RH_GetNextRandom2D(state)) / float2(dim);

        float uvPdf = pTexel * float(dim.x * dim.y);
        float invUvPdf = (uvPdf > 0.f) ? (1.f / uvPdf) : 0.f;

        gPresampledEnvData[index] = uint2(packUv16(uv), asuint(invUvPdf));
    }

    int2 getPrevPixel(uint2 pixel, RH_RandomSamplerState sg)
    {
        float2 curr = float2(pixel);
        float2 motion = mvec[pixel] * gFrameDim;
        motion += RH_GetNextRandom2D(sg) - 0.5;
        float2 prev = curr + motion;
        return int2(round(prev));
    }

    bool areSurfacesCompatible(const RTAB_Surface a, const RTAB_Surface b)
    {
        if (!a.valid || !b.valid) return false;
        if (dot(a.normal, b.normal) < 0.85f) return false;

        float3 dP = b.position - a.position;
        if (abs(dot(dP, a.normal)) > 0.01f) return false;
        return true;
    }
    /** Compares two values and returns true if their relative difference is lower than the threshold.
        Zero or negative threshold makes test always succeed, not fail.
    */
    bool compareRelativeDifference(float reference, float candidate, float threshold) {
        return (threshold <= 0) || abs(reference - candidate) <= threshold * max(reference, candidate);
    }
    /** judge by normal & depth
     */
    bool isValidNeighbor(
        float3 n1, float3 n2, float depth1, float depth2,
        float normalThreshold,
        float depthThreshold
    ) {
        return (dot(n1.xyz, n2.xyz) >= normalThreshold) && compareRelativeDifference(depth1, depth2, depthThreshold);
    }

    void init(uint2 pixel)
    {
        if (any(pixel >= gFrameDim)) return;

        uint pixelIndex = pixel.y * gFrameDim.x + pixel.x;
        RTAB_Surface surf;
        bool validSurface = buildSurfaceData(pixel, surf);
        surfaceData[getSurfaceIndex(pixel, currentSurfaceBufferIndex)] = surf.pack();

        RH_RandomSamplerState randomState = RH_InitRandomState(pixel, gFrameIndex, 0);
        RH_RandomSamplerState coherentState = RH_InitRandomState(pixel / 16u, gFrameIndex, 4);

        Reservoir r = Reservoir();
        if (!validSurface || totalLightCount == 0)
        {
            gReservoir[pixelIndex] = r;
            return;
        }

        uint numMisSamples = gLocalRIS_M + gEnvRIS_M + gBrdfRIS_M;
        float localMisWeight = (numMisSamples > 0u) ? (float(gLocalRIS_M) / float(numMisSamples)) : 0.f;
        float envMisWeight = (numMisSamples > 0u) ? (float(gEnvRIS_M) / float(numMisSamples)) : 0.f;
        float brdfMisWeight = (numMisSamples > 0u) ? (float(gBrdfRIS_M) / float(numMisSamples)) : 0.f;
        uint localTileIndex = (gPresampledTileCount > 0u) ? min((uint)(RH_GetNextRandom(coherentState) * gPresampledTileCount), gPresampledTileCount - 1u) : 0u;
        uint envTileIndex = (gPresampledTileCount > 0u) ? min((uint)(RH_GetNextRandom(coherentState) * gPresampledTileCount), gPresampledTileCount - 1u) : 0u;

        Reservoir localReservoir = buildCategoryReservoir(surf, gLocalRIS_M, localMisWeight, float(max(numMisSamples, 1u)), randomState, 0u, localTileIndex);
        Reservoir infiniteReservoir = buildCategoryReservoir(surf, gInfiniteRIS_M, 1.f, float(max(gInfiniteRIS_M, 1u)), randomState, 1u, 0u);
        Reservoir envReservoir = buildCategoryReservoir(surf, gEnvRIS_M, envMisWeight, float(max(numMisSamples, 1u)), randomState, 2u, envTileIndex);
        Reservoir brdfReservoir = buildBrdfReservoir(surf, gBrdfRIS_M, localMisWeight, envMisWeight, brdfMisWeight, float(max(numMisSamples, 1u)), randomState);

        RTDI_CombineReservoirs(r, localReservoir, 0.5f, localReservoir.targetPdf);
        RTDI_CombineReservoirs(r, infiniteReservoir, RH_GetNextRandom(randomState), infiniteReservoir.targetPdf);
        RTDI_CombineReservoirs(r, envReservoir, RH_GetNextRandom(randomState), envReservoir.targetPdf);
        RTDI_CombineReservoirs(r, brdfReservoir, RH_GetNextRandom(randomState), brdfReservoir.targetPdf);
        RTDI_FinalizeResampling(r, 1.0f, 1.0f);
        r.M = (RTDI_IsValidReservoir(r) || localReservoir.M > 0.f || infiniteReservoir.M > 0.f || envReservoir.M > 0.f || brdfReservoir.M > 0.f) ? 1.f : 0.f;

        gReservoir[pixelIndex] = r;
    }

    void visibility(uint2 pixel)
    {
        if (any(pixel >= gFrameDim)) return;

        uint pixelIndex = pixel.y * gFrameDim.x + pixel.x;
        Reservoir r = gReservoir[pixelIndex];
        if (!RTDI_IsValidReservoir(r)) return;

        RTAB_Surface surf;
        if (!loadSurface(pixel, currentSurfaceBufferIndex, true, surf))
        {
            gReservoir[pixelIndex] = Reservoir();
            return;
        }

        PolymorphicLightSample ls;
        if (!samplePolymorphicLight(surf, r.lightID, r.sampleUV, ls))
        {
            discardReservoirSample(r);
            gReservoir[pixelIndex] = r;
            return;
        }

        float tMax = (ls.distance >= FLT_MAX * 0.5f) ? FLT_MAX : max(ls.distance - 2e-3f, 0.f);
        SceneRayQuery<0> rayQuery;
        bool visible = rayQuery.traceVisibilityRay(Ray(surf.position, ls.direction, 0.f, tMax),RAY_FLAG_CULL_NON_OPAQUE);
        if (!visible)
        {
            discardReservoirSample(r);
        }

        gReservoir[pixelIndex] = r;
    }

    void temporalReuse(uint2 pixel, uint maxHistoryLength, float samplingRadius)
    {

        if (any(pixel >= gFrameDim)) return;
        RH_RandomSamplerState randomState = RH_InitRandomState(pixel, gFrameIndex, 2);
        uint pixelIndex = pixel.y * gFrameDim.x + pixel.x;
        RTAB_Surface surf;
        if (!loadSurface(pixel, currentSurfaceBufferIndex, true, surf))
        {
            gReservoir[pixelIndex] = Reservoir();
            return;
        }
        Reservoir baseR = gReservoir[pixelIndex];

        Reservoir state = Reservoir();
        RTDI_CombineReservoirs(state, baseR, 0.5f, baseR.targetPdf);
        uint historyLimit = min(RTDI_ReservoirMaxM, maxHistoryLength * baseR.M);

        int2 prevPixel = getPrevPixel(pixel, randomState);
        int2 spatialOffset = int2(0, 0);
        bool findNeighbor = false;
        float radius = samplingRadius;
        float3 motion = float3(mvec[pixel], 0.f); //TODO handle linear z
        float expectedPrevLinearDepth = RTAB_GetSurfaceLinearDepth(surf) + motion.z;
        RTAB_Surface tempSurface;
        for (int i = 0; i < 9; i++) {
            int2 offset = int2(0, 0);
            if (i > 0) {
                offset.x = int((RH_GetNextRandom(randomState) - 0.5) * radius);
                offset.y = int((RH_GetNextRandom(randomState) - 0.5) * radius);
            }
            int2 idx = prevPixel + offset;
            bool validSurface = loadSurface(idx, prevSurfaceBufferIndex, true, tempSurface);
            if (!validSurface) continue;
            if (!isValidNeighbor(
                    surf.normal, tempSurface.normal,
                    expectedPrevLinearDepth, RTAB_GetSurfaceLinearDepth(tempSurface),
                    normalThreshold, depthThreshold
            )) continue;

            findNeighbor = true;
            spatialOffset = idx - prevPixel;
            prevPixel = idx;
            break;
        }

        if (findNeighbor) {
            uint prevPixelIndex = uint(prevPixel.y) * gFrameDim.x + uint(prevPixel.x);
            Reservoir prevSample = gPrevReservoir[prevPixelIndex];
            prevSample.M = min(prevSample.M, historyLimit);
            float currentPdf = 0.f;
            if (RTDI_IsValidReservoir(prevSample))
            {
                PolymorphicLightSample ls;
                if (samplePolymorphicLight(surf, prevSample.lightID, prevSample.sampleUV, true, ls))
                {
                    currentPdf = calcTargetPdf(surf, ls.direction, ls.Li);
                }
            }
            RTDI_CombineReservoirs(state, prevSample, RH_GetNextRandom(randomState), currentPdf);
        }
        RTDI_FinalizeResampling(state, 1.0, state.M);
        //TODO bias correction
        gReservoir[pixelIndex] = state;
    }
    /** spatial resampling
    */
    void spatialReuse(uint2 pixel, uint numSamples, float samplingRadius)
    {
        if (any(pixel >= gFrameDim)) return;
        uint centerPixelIndex = pixel.y * gFrameDim.x + pixel.x;
        RH_RandomSamplerState rng = RH_InitRandomState(pixel, gFrameIndex, 3);
        Reservoir centerSample = gPrevReservoir[centerPixelIndex];
        if (!RTDI_IsValidReservoir(centerSample))
        {
            gReservoir[centerPixelIndex] = Reservoir();
            return;
        }
        RTAB_Surface centerSurface;
        if (!loadSurface(pixel, currentSurfaceBufferIndex, true, centerSurface))
        {
            gReservoir[centerPixelIndex] = Reservoir();
            return;
        }

        Reservoir state = Reservoir();
        RTDI_CombineReservoirs(state, centerSample, 0.5f, centerSample.targetPdf);

        // TODO When the length of history is less than targetHistoryLength
        // customize the number of samples to select.
        uint numSpatialSamples = min(numSamples, 32u);
        int i = 0;
        RTAB_Surface neighborSurface;
        uint startIdx = uint(RH_GetNextRandom(rng) * (float)gNeighborOffsetMask);
        for (i = 0; i < numSpatialSamples; i++) {
            uint sampleIdx = (startIdx + i) & gNeighborOffsetMask;
            int2 spatialOffset = int2(0, 0);
            spatialOffset = int2(float2(gNeighborOffsetBuffer[sampleIdx].xy) * samplingRadius);
            int2 idx = int2(pixel) + spatialOffset;
            bool isValidSurface = loadSurface(idx, currentSurfaceBufferIndex, true, neighborSurface);
            if (!isValidSurface) continue;
            // TODO outside Threshold
            if (!isValidNeighbor(
                centerSurface.normal,neighborSurface.normal,
                RTAB_GetSurfaceLinearDepth(centerSurface), RTAB_GetSurfaceLinearDepth(neighborSurface),
                normalThreshold, depthThreshold))
                continue;
            uint neighborIndex = idx.y * gFrameDim.x + idx.x;
            Reservoir neighborSample = gPrevReservoir[neighborIndex];
            if (!RTDI_IsValidReservoir(neighborSample))
                continue;
            float neighborTargetPdf = 0.f;
            PolymorphicLightSample candidateLightSample;
            if (samplePolymorphicLight(centerSurface, neighborSample.lightID, neighborSample.sampleUV, true, candidateLightSample))
            {
                neighborTargetPdf = calcTargetPdf(centerSurface, candidateLightSample.direction, candidateLightSample.Li);
            }
            RTDI_CombineReservoirs(state, neighborSample, RH_GetNextRandom(rng), neighborTargetPdf);
        }
        RTDI_FinalizeResampling(state, 1.0f, state.M);
        gReservoir[centerPixelIndex] = state;
    }

    void shadeColor(uint2 pixel)
    {
        if (any(pixel >= gFrameDim)) return;
        uint pixelIndex = pixel.y * gFrameDim.x + pixel.x;
        Reservoir r = gReservoir[pixelIndex];
        ShadingData sd;
        if (loadShadingData(pixel, gFrameDim, gScene.camera, vbuffer, sd))
        {

            RTAB_Surface surf;
            bool validSurface = loadSurface(pixel, currentSurfaceBufferIndex, true, surf);
            let lod = ExplicitLodTextureSampler(0.0f);
            let hints = getMaterialInstanceHints();
            let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
            float3 diffuse = {};
            float3 specular = {};
            if (!validSurface)
            {
                color[pixelIndex] = float4(0, 0, 0, 1);
                return;
            }
            PolymorphicLightSample ls;
            if (RTDI_IsValidReservoir(r) && samplePolymorphicLight(surf, r.lightID, r.sampleUV, ls)) {
                float tMax = (ls.distance >= FLT_MAX * 0.5f) ? FLT_MAX : max(ls.distance - 2e-3f, 0.f);
                SceneRayQuery<1> rayQuery;
                bool visible = rayQuery.traceVisibilityRay(Ray(surf.position, ls.direction, 0.f, tMax));
                if (visible) {
                    TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(pixel, gFrameIndex);
                    sd.mtl.setActiveLobes((uint)LobeType::DiffuseReflection);
                    diffuse = mi.eval(sd, ls.direction, sg) * ls.Li;
                    sd.mtl.setActiveLobes((uint)LobeType::SpecularReflection);
                    specular = mi.eval(sd, ls.direction, sg) * ls.Li;
                }
            }
            let props = mi.getProperties(sd);

            color[pixel] = float4((diffuse + specular) * r.wsum + props.emission, 1);
            return;
        } else {
            float3 dir = gScene.camera.computeRayPinhole(pixel, gFrameDim).dir;
            color[pixel] = float4(gScene.envMap.eval(dir),1);
        }
    }
};

ParameterBlock<RTDI> gRTDI;
