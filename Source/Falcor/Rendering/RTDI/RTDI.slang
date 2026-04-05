import Utils.Math.HashUtils;
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
    uint gRIS_M;
    uint gPresampledTileCount;
    uint gPresampledTileSize;
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

    // Internal algorithm resources.
    RWStructuredBuffer<Reservoir> gReservoir;
    StructuredBuffer<Reservoir> gPrevReservoir;

    StructuredBuffer<PackedPolymorphicLight> gLightInfo;
    RWStructuredBuffer<PackedSurfaceData> surfaceData;

    RWStructuredBuffer<uint> gPresampledLightIndex;
    RWStructuredBuffer<uint2> gPresampledEnvData;

    Texture2D<PackedHitInfo> vbuffer;
    Texture2D<float2> mvec;
    Texture2D<float> localLightPdfTexture;
    Texture2D<float> envLightLuminanceTexture;
    Texture2D<float> envLightPdfTexture;
    RWTexture2D<float4> color;

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
            float3 toEye = gScene.camera.data.posW - asfloat(surfaceData[idx].position);
            if (dot(toEye, toEye) > 0.f) viewDir = normalize(toEye);
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

        float roughness = max(surf.roughness, 0.02f);
        float specular = max(surf.specular, 0.f);
        float D = evalNdfGGX(roughness * roughness, NdotH);
        float G = evalMaskingSmithGGXSeparable(roughness, NdotL, NdotV);
        float F = specular < 1e-8f ? 0.f : (evalFresnelSchlick(specular, 1.f, LdotH) / max(specular, 1e-6f));

        float diffuseBRDF = NdotL * (1.f / PI);
        float specularBRDF = (D * G * F) / max(4.f * NdotV, 1e-6f);
        float fullBRDF = max(0.f, lerp(specularBRDF, diffuseBRDF, surf.diffuseProb));
        return fullBRDF * max(Li.x, 0.f);
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

        float sampleFromPdfMipTreeWithProb(Texture2D<float> tex, inout RH_RandomSamplerState randomState, out uint2 coord)
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

    float pdfTexelFromMipTree(Texture2D<float> tex, uint2 texel)
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
            localAvail = max(localLightPdfTexture.Load(int3(0, 0, int(mipCount - 1))), 0.f) > 0.f;
        }

        if (envLightIndex < totalLightCount)
        {
            uint mipCount;
            uint2 dim;
            envLightPdfTexture.GetDimensions(0, dim.x, dim.y, mipCount);
            envAvail = max(envLightPdfTexture.Load(int3(0, 0, int(mipCount - 1))), 0.f) > 0.f;
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
        float texelValue = max(localLightPdfTexture.Load(int3(texel, 0)), 0.f);

        int lastMip = max(0, int(floor(log2(max(dim.x, dim.y)))) - 1);
        float avg0 = max(localLightPdfTexture.Load(int3(0, 0, lastMip)), 0.f);
        float avg1 = max(localLightPdfTexture.Load(int3(1, 0, lastMip)), 0.f);
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
        float texelValue = max(envLightPdfTexture.Load(int3(texel, 0)), 0.f);

        int lastMip = max(0, int(floor(log2(max(dim.x, dim.y)))) - 1);
        float avg0 = max(envLightPdfTexture.Load(int3(0, 0, lastMip)), 0.f);
        float avg1 = max(envLightPdfTexture.Load(int3(1, 0, lastMip)), 0.f);
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

    float calcProposalPdf(int lightID, PolymorphicLightSample ls)
    {
        float pCategory = calcCategorySelectionPdf(lightID);
        if (pCategory <= 0.f) return 0.f;

        float pSelect = 0.f;
        if (lightID >= 0 && uint(lightID) < firstInfiniteAnalyticLight)
        {
            pSelect = pCategory * calcLocalLightSourcePdf(lightID);
        }
        else if (lightID >= 0 && uint(lightID) < envLightIndex)
        {
            uint count = envLightIndex - firstInfiniteAnalyticLight;
            pSelect = pCategory * ((count > 0) ? (1.f / float(count)) : 0.f);
        }
        else if (lightID >= 0 && uint(lightID) == envLightIndex)
        {
            pSelect = pCategory * calcEnvironmentMapSamplingPdf(ls.direction);
        }

        if (pSelect <= 0.f) return 0.f;

        float pCond = max(ls.pdf, 1e-6f);
        return pSelect * pCond;
    }

    float calcw(RTAB_Surface surf, int lightID, float2 sampleUV, out float ph)
    {
        ph = 0.f;
        if (lightID < 0 || uint(lightID) >= totalLightCount) return 0.f;

        PolymorphicLightSample ls;
        if (!samplePolymorphicLight(surf, lightID, sampleUV, true, ls)) return 0.f;

        float p = calcProposalPdf(lightID, ls);
        if (p <= 0.f) return 0.f;

        ph = calcTargetPdf(surf, ls.direction, ls.Li);
        return ph / max(p, 1e-6f);
    }

    uint sampleFallbackLight(inout RH_RandomSamplerState randomState, uint lightCount)
    {
        uint p = min((uint)(RH_GetNextRandom(randomState) * lightCount), lightCount - 1);
        return p;
    }

    float loadPdfSafe(Texture2D<float> tex, uint mip, int2 p)
    {
        uint2 dim;
        uint mipCount;
        tex.GetDimensions(mip, dim.x, dim.y, mipCount);
        if (p.x < 0 || p.y < 0 || p.x >= int(dim.x) || p.y >= int(dim.y)) return 0.f;
        return max(tex.Load(int3(p, mip)), 0.f);
    }

    uint2 sampleFromPdfMipTree(Texture2D<float> tex, inout RH_RandomSamplerState randomState)
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

    void presampleLocalLights(uint index)
    {
        uint total = max(gPresampledTileCount * gPresampledTileSize, 1u);
        if (index >= total) return;

        if (totalLightCount == 0)
        {
            gPresampledLightIndex[index] = 0;
            return;
        }

        RH_RandomSamplerState state = RH_InitRandomState(uint2(index,index),gFrameIndex,1);
        uint sampledLight;
        float2 sampledUV;
        sampleLightByPdf(state, sampledLight, sampledUV);
        gPresampledLightIndex[index] = sampledLight;
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

    uint2 getPrevPixel(uint2 pixel, RH_RandomSamplerState sg)
    {
        float2 curr = float2(pixel);
        float2 motion = mvec[pixel] * gFrameDim;
        motion += RH_GetNextRandom2D(sg) - 0.5;
        float2 prev = curr + motion;
        int2 prevPix = int2(round(prev));

        // if (prevPix.x < 0 || prevPix.y < 0 || prevPix.x >= gFrameDim.x || prevPix.y >= gFrameDim.y) return pixel;
        return uint2(prevPix);
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

        static const int kMaxSampleCount = 64;
        int n = clamp(gRIS_M, 0, kMaxSampleCount);
        uint lightCount = totalLightCount;
        RH_RandomSamplerState randomState = RH_InitRandomState(pixel, gFrameIndex, 0);

        Reservoir r = Reservoir();
        if (!validSurface || lightCount == 0 || n == 0)
        {
            gReservoir[pixelIndex] = r;
            return;
        }

        [unroll]
        for (int i = 0; i < kMaxSampleCount; ++i)
        {
            if (i >= n) break;

            int p = -1;
            float2 uv = RH_GetNextRandom2D(randomState);
            uint totalPresampled = max(gPresampledTileCount * gPresampledTileSize, 1u);
            if (gPresampledTileCount > 0 && gPresampledTileSize > 0)
            {
                uint pixelLinear = pixel.y * gFrameDim.x + pixel.x;
                uint tileId = (pixelLinear + (uint)i + gFrameIndex * 17u) % gPresampledTileCount;
                uint slot = min((uint)(RH_GetNextRandom(randomState) * gPresampledTileSize), gPresampledTileSize - 1);
                uint entry = min(tileId * gPresampledTileSize + slot, totalPresampled - 1);
                p = int(min(gPresampledLightIndex[entry], lightCount - 1));

                if (p >= 0 && uint(p) == envLightIndex)
                {
                    uint2 envSample = gPresampledEnvData[entry];
                    if (envSample.y != 0u) uv = unpackUv16(envSample.x);
                    else
                    {
                        float2 envUv;
                        if (sampleEnvFromPdfTexture(randomState, envUv)) uv = envUv;
                    }
                }
            }
            else
            {
                uint sampledLight = 0;
                sampleLightByPdf(randomState, sampledLight, uv);
                p = int(sampledLight);
            }

            if (p < 0 || uint(p) >= totalLightCount) continue;

            float ph = 0.f;
            float wi = calcw(surf, p, uv, ph);
            r.update(p, uv, wi, ph, 1.f, randomState);
        }

        if (r.M > 0 && r.targetPdf > 0.f)
        {
            r.W = r.wsum / (float(r.M) * r.targetPdf);
        }
        else r.W = 0.f;

        gReservoir[pixelIndex] = r;
    }

    void visibility(uint2 pixel)
    {
        if (any(pixel >= gFrameDim)) return;

        uint pixelIndex = pixel.y * gFrameDim.x + pixel.x;
        Reservoir r = gReservoir[pixelIndex];
        if (!RTDI_IsValidReservoir(r) || r.W <= 0.f) return;

        RTAB_Surface surf;
        if (!loadSurface(pixel, currentSurfaceBufferIndex, true, surf))
        {
            gReservoir[pixelIndex] = Reservoir();
            return;
        }

        PolymorphicLightSample ls;
        if (!samplePolymorphicLight(surf, r.lightID, r.sampleUV, ls))
        {
            gReservoir[pixelIndex] = Reservoir();
            return;
        }

        float tMax = (ls.distance >= FLT_MAX * 0.5f) ? FLT_MAX : max(ls.distance - 2e-3f, 0.f);
        SceneRayQuery<1> rayQuery;
        bool visible = rayQuery.traceVisibilityRay(Ray(surf.position, ls.direction, 0.f, tMax));
        if (!visible)
        {
            r.W = 0.f;
            r.wsum = 0.f;
            r.targetPdf = 0.f;
        }

        gReservoir[pixelIndex] = r;
    }

    void temporalReuse(uint2 pixel, uint maxHistoryLength)
    {
        if (any(pixel >= gFrameDim)) return;
        RH_RandomSamplerState randomState = RH_InitRandomState(pixel, gFrameIndex, 2);
        uint pixelIndex = pixel.y * gFrameDim.x + pixel.x;
        RTAB_Surface surf;
        loadSurface(pixel, currentSurfaceBufferIndex, true, surf);
        Reservoir baseR = gReservoir[pixelIndex];
        Reservoir state = Reservoir();
        RTDI_CombineReservoirs(state, baseR, RH_GetNextRandom(randomState), baseR.targetPdf);
        uint historyLimit = min(RTDI_ReservoirMaxM, maxHistoryLength * baseR.M);

        uint2 prevPixel = getPrevPixel(pixel, randomState);
        int2 spatialOffset = int2(0, 0);
        bool findNeighbor = false;
        float radius = 4;
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
            uint prevPixelIndex = prevPixel.y * gFrameDim.x + prevPixel.x;
            Reservoir prevSample = gPrevReservoir[prevPixelIndex];
            if (RTDI_IsValidReservoir(prevSample))
            {
                prevSample.M = min(prevSample.M, historyLimit);
                PolymorphicLightSample ls;
                samplePolymorphicLight(surf, prevSample.lightID, prevSample.sampleUV, ls);
                float currentPdf = calcTargetPdf(surf, ls.direction, ls.Li);
                bool sampleSelected = RTDI_CombineReservoirs(state, prevSample, RH_GetNextRandom(randomState), currentPdf);
            }
        }
        RTDI_FinalizeResampling(state, 1.0, state.M);
        //TODO bias correction
        gReservoir[pixelIndex] = state;
    }
    /** spatial resampling
    */
    void spatialReuse(uint2 pixel, float numSamples, float samplingRadius)
    {
        if (any(pixel >= gFrameDim)) return;
        uint centerPixelIndex = pixel.y * gFrameDim.x + pixel.x;
        RH_RandomSamplerState rng = RH_InitRandomState(pixel, gFrameIndex, 3);
        Reservoir state = Reservoir();
        Reservoir centerSample = gPrevReservoir[centerPixelIndex];
        PolymorphicLight selectedLight = {};
        if (RTDI_IsValidReservoir(centerSample)) {
            selectedLight = PolymorphicLight(gLightInfo[uint(centerSample.lightID)]);
        }
        RTDI_CombineReservoirs(state, centerSample, RH_GetNextRandom(rng), centerSample.targetPdf);
        RTAB_Surface centerSurface;
        bool centerSurfaceValid = loadSurface(pixel, currentSurfaceBufferIndex, true, centerSurface);
        // TODO When the length of history is less than targetHistoryLength
        // customize the number of samples to select.
        uint numSpatialSamples = min(numSamples, 32);
        uint cachedResult = 0;
        int i = 0;
        RTAB_Surface neighborSurface;
        for (i = 0; i < numSpatialSamples; i++) {
            int2 spatialOffset = int2(0, 0);
            spatialOffset = int2(RH_GetNextRandom2D(rng) * samplingRadius);
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
            if (RTDI_IsValidReservoir(neighborSample)) {
                PolymorphicLightSample candidateLightSample;
                samplePolymorphicLight(centerSurface, neighborSample.lightID, neighborSample.sampleUV, candidateLightSample);
                bool selectNeighbor = RTDI_CombineReservoirs(state, neighborSample, RH_GetNextRandom(rng), calcTargetPdf(centerSurface, candidateLightSample.direction, candidateLightSample.Li));
            }
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

            if (!validSurface || !RTDI_IsValidReservoir(r) || r.W <= 0.f)
            {
                color[pixel] = float4(0, 0, 0, 1);
                return;
            }
            PolymorphicLightSample ls;
            if (!samplePolymorphicLight(surf, r.lightID, r.sampleUV, ls))
            {
                color[pixel] = float4(0, 0, 0, 1);
                return;
            }

            float tMax = (ls.distance >= FLT_MAX * 0.5f) ? FLT_MAX : max(ls.distance - 2e-3f, 0.f);
            SceneRayQuery<1> rayQuery;
            bool visible = rayQuery.traceVisibilityRay(Ray(surf.position, ls.direction, 0.f, tMax));
            if (!visible)
            {
                color[pixel] = float4(0, 0, 0, 1);
                return;
            }
            let lod = ExplicitLodTextureSampler(0.0f);
            let hints = getMaterialInstanceHints();
            let mi = gScene.materials.getMaterialInstance(sd, lod, hints);

            TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(pixel, gFrameIndex);
            sd.mtl.setActiveLobes((uint)LobeType::DiffuseReflection);
            float3 diffuse = mi.eval(sd, ls.direction, sg) * ls.Li;
            sd.mtl.setActiveLobes((uint)LobeType::SpecularReflection);
            float3 specular = mi.eval(sd, ls.direction, sg) * ls.Li;

            let props = mi.getProperties(sd);
            color[pixel] = float4((diffuse + specular)*r.W + props.emission, 1);
            return;
        } else {
            float3 dir = gScene.camera.computeRayPinhole(pixel, gFrameDim).dir;
            color[pixel] = float4(gScene.envMap.eval(dir),1);
        }
    }
};

ParameterBlock<RTDI> gRTDI;
