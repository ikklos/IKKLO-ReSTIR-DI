import Utils.Math.HashUtils;
import Utils.Math.ShadingFrame;
import Utils.Geometry.IntersectionHelpers;
import Utils.Color.ColorHelpers;
import Utils.Sampling.TinyUniformSampleGenerator;

import Scene.RaytracingInline;
import Scene.Scene;
import RenderPasses.ReSTIR_DI.LoadShadingData;

import Utils.Math.Ray;
#include "RandomHelper.slangh"
#include "rtxdi/RtxdiMath.hlsli"
#define RTXDI_POLYMORPHIC_LIGHT_SAMPLING
#include "SurfaceData.slang"
#include "PolymorphicLight.slang"

#define PI 3.14159265358979323846
static const uint kMaxReservoirAge = 32u;

struct Reservoir
{
    uint lightID = 0;
    float2 sampleUV = float2(0, 0);

    float W = 0.0;
    float wsum = 0.0;
    uint M = 0;
    uint age = 0;

    [mutating] void update(uint light, float2 uv, float w, inout uint state)
    {
        wsum = wsum + w;
        M = M + 1;
        if (rand01(state) < (w / max(wsum, 1e-6f)))
        {
            lightID = light;
            sampleUV = uv;
        }
    }
};

struct RTDI
{
    // Runtime parameters.
    uint gRIS_M;
    uint gPresampledTileCount;
    uint gPresampledTileSize;
    uint2 gFrameDim;
    uint gFrameIndex;
    uint gRandomSeed;
    uint gDebugShowWExplosion;
    float gDebugWThreshold;

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

    StructuredBuffer<float> gAliasProb;
    StructuredBuffer<uint> gAliasIndex;
    StructuredBuffer<float> gLightPmf;
    StructuredBuffer<PackedPolymorphicLight> gLightInfo;
    RWStructuredBuffer<PackedSurfaceData> surfaceData;

    RWStructuredBuffer<uint> gPresampledLightIndex;

    Texture2D<PackedHitInfo> vbuffer;
    Texture2D<float2> mvec;
    Texture2D<float> localLightPdfTexture;
    Texture2D<float> envLightLuminanceTexture;
    Texture2D<float> envLightPdfTexture;
    RWTexture2D<float4> color;

    void markExplosionStage(inout Reservoir r, uint stage)
    {
        if (r.age == 0 && r.W > gDebugWThreshold)
            r.age = stage;
    }

    float3 getExplosionDebugColor(uint stage, float w)
    {
        float intensity = saturate(log2(max(w, 1e-6f)) / 8.f);
        if (stage == 1) return float3(intensity, 0.f, 0.f);
        if (stage == 2) return float3(intensity, intensity, 0.f);
        if (stage == 3) return float3(intensity, 0.f, intensity);
        if (stage == 4) return float3(0.f, intensity, intensity);
        return float3(intensity, intensity, intensity);
    }

    uint getSurfaceIndex(uint2 pixel, uint bufferIndex)
    {
        return pixel.x + pixel.y * gFrameDim.x + bufferIndex * (gFrameDim.x * gFrameDim.y);
    }

    bool buildSurfaceData(uint2 pixel, out SurfaceData sd)
    {
        sd = SurfaceData();
        ShadingData hit;
        if (!loadShadingData(pixel, gFrameDim, gScene.camera, vbuffer, hit))
            return false;

        const uint hints = getMaterialInstanceHints();
        let lod = ExplicitLodTextureSampler(0.f);
        let mi = gScene.materials.getMaterialInstance(hit, lod, hints);
        let props = mi.getProperties(hit);

        sd.position = hit.posW;
        sd.viewDepth = length(hit.posW - gScene.camera.data.posW);
        sd.normal = props.guideNormal;
        sd.viewDir = normalize(hit.V);
        sd.diffuse = saturate(luminance(props.diffuseReflectionAlbedo));
        sd.specular = saturate(luminance(props.specularReflectance));
        sd.roughness = saturate(props.roughness);
        sd.diffuseProb = sd.diffuse / max(sd.diffuse + sd.specular, 1e-6f);

        return true;
    }

    bool loadSurface(uint2 pixel, uint bufferIndex, bool withViewDir, out SurfaceData sd)
    {
        sd = SurfaceData();
        if (any(pixel >= gFrameDim)) return false;

        uint idx = getSurfaceIndex(pixel, bufferIndex);
        float3 viewDir = float3(0.f, 0.f, 1.f);
        if (withViewDir)
        {
            float3 toEye = gScene.camera.data.posW - asfloat(surfaceData[idx].position);
            if (dot(toEye, toEye) > 0.f) viewDir = normalize(toEye);
        }

        sd = SurfaceData(surfaceData[idx], viewDir);
        return sd.valid;
    }

    float evalNdfGGXLocal(float alpha2, float NdotH)
    {
        float d = NdotH * NdotH * (alpha2 - 1.f) + 1.f;
        return alpha2 / max(PI * d * d, 1e-6f);
    }

    float evalMaskingSchlickGGX(float roughness, float NdotL, float NdotV)
    {
        float k = (roughness + 1.f);
        k = (k * k) * 0.125f;
        float gL = NdotL / max(NdotL * (1.f - k) + k, 1e-6f);
        float gV = NdotV / max(NdotV * (1.f - k) + k, 1e-6f);
        return gL * gV;
    }

    float fresnelSchlickScalar(float F0, float LdotH)
    {
        float f = pow(1.f - LdotH, 5.f);
        return F0 + (1.f - F0) * f;
    }

    float calcTargetPdf(SurfaceData surf, float3 wi, float3 Li)
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
        float alpha = roughness * roughness;
        float D = evalNdfGGXLocal(alpha * alpha, NdotH);
        float G = evalMaskingSchlickGGX(roughness, NdotL, NdotV);

        float diffuseL = max(surf.diffuse, 0.f);
        float specularL = max(surf.specular, 0.f);
        float F0 = saturate(specularL);
        float F = fresnelSchlickScalar(F0, LdotH);

        float diffuseProb = diffuseL / max(diffuseL + specularL, 1e-6f);
        float diffuseBRDF = diffuseL * NdotL * (1.f / PI);
        float specularBRDF = (D * G * F) / max(4.f * NdotL * NdotV, 1e-6f);
        float fullBRDF = max(0.f, lerp(specularBRDF, diffuseBRDF, diffuseProb));
        return fullBRDF * max(luminance(Li), 0.f);
    }

            bool samplePolymorphicLight(SurfaceData surf, uint lightID, float2 sampleUV, out PolymorphicLightSample ls)
    {
        ls = {};
        if (lightID >= totalLightCount) return false;
        PolymorphicLight light = PolymorphicLight(gLightInfo[lightID]);
        ls = light.calcSample(surf.position, sampleUV, false);
        return ls.valid;
    }

        float sampleFromPdfMipTreeWithProb(Texture2D<float> tex, inout uint randomState, out uint2 coord)
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
                uint q = min((uint)(rand01(randomState) * 4.f), 3u);
                off = uint2(q & 1u, (q >> 1u) & 1u);
                prob *= 0.25f;
            }
            else
            {
                float u = rand01(randomState) * sum;
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

    float calcLightSelectionPdf(uint lightID)
    {
        bool localAvail, infAvail, envAvail;
        getCategoryAvailability(localAvail, infAvail, envAvail);

        uint categoryCount = (localAvail ? 1u : 0u) + (infAvail ? 1u : 0u) + (envAvail ? 1u : 0u);
        if (categoryCount == 0u) return (totalLightCount > 0) ? (1.f / float(totalLightCount)) : 0.f;

        float pCategory = 1.f / float(categoryCount);

        if (lightID < firstInfiniteAnalyticLight)
        {
            if (!localAvail) return 0.f;
            uint2 texel = RTXDI_LinearIndexToZCurve(lightID);
            return pCategory * pdfTexelFromMipTree(localLightPdfTexture, texel);
        }

        if (lightID < envLightIndex)
        {
            if (!infAvail) return 0.f;
            uint count = envLightIndex - firstInfiniteAnalyticLight;
            return pCategory * (1.f / float(count));
        }

        if (lightID == envLightIndex)
        {
            if (!envAvail) return 0.f;
            return pCategory;
        }

        return 0.f;
    }

    float calcEnvDirectionalPdf(float2 sampleUV)
    {
        uint2 dim;
        uint mipCount;
        envLightPdfTexture.GetDimensions(0, dim.x, dim.y, mipCount);
        if (dim.x == 0 || dim.y == 0) return 0.f;

        float2 uv = clamp(sampleUV, float2(0.f), float2(1.f - 1e-6f));
        uint2 texel = min((uint2)(uv * float2(dim)), dim - 1);
        float pTexel = pdfTexelFromMipTree(envLightPdfTexture, texel);

        float uvPdf = pTexel * float(dim.x * dim.y);
        float theta = (uv.y - 0.5f) * PI;
        float relativeSolidAngle = max(cos(theta), 1e-6f);
        return uvPdf / max(2.f * PI * PI * relativeSolidAngle, 1e-6f);
    }

    float calcProposalPdf(uint lightID, float2 sampleUV, PolymorphicLightSample ls)
    {
        float pSelect = calcLightSelectionPdf(lightID);
        if (pSelect <= 0.f) return 0.f;

        float pCond = (lightID == envLightIndex) ? calcEnvDirectionalPdf(sampleUV) : max(ls.pdf, 1e-6f);
        return pSelect * pCond;
    }

    float calcw(SurfaceData surf, uint lightID, float2 sampleUV)
    {
        if (lightID >= totalLightCount) return 0.f;

        PolymorphicLightSample ls;
        if (!samplePolymorphicLight(surf, lightID, sampleUV, ls)) return 0.f;

        float p = calcProposalPdf(lightID, sampleUV, ls);
        if (p <= 0.f) return 0.f;

        float ph = calcTargetPdf(surf, ls.direction, ls.Li);
        return ph / max(p, 1e-6f);
    }

    uint sampleFallbackLight(inout uint randomState, uint lightCount)
    {
        uint p = min((uint)(rand01(randomState) * lightCount), lightCount - 1);
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

    uint2 sampleFromPdfMipTree(Texture2D<float> tex, inout uint randomState)
    {
        uint2 coord;
        sampleFromPdfMipTreeWithProb(tex, randomState, coord);
        return coord;
    }


    bool sampleLocalLightFromPdfTexture(inout uint randomState, out uint lightID)
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

    bool sampleEnvFromPdfTexture(inout uint randomState, out float2 uv)
    {
        uv = float2(0.5f, 0.5f);
        if (envLightIndex >= totalLightCount) return false;

        uint2 dim;
        uint mipCount;
        envLightPdfTexture.GetDimensions(0, dim.x, dim.y, mipCount);
        if (dim.x == 0 || dim.y == 0) return false;

        uint2 texel = sampleFromPdfMipTree(envLightPdfTexture, randomState);
        uv = (float2(texel) + rand2(randomState)) / float2(dim);
        return true;
    }

    bool sampleInfiniteAnalyticFromPmf(inout uint randomState, out uint lightID)
    {
        lightID = 0;
        uint begin = firstInfiniteAnalyticLight;
        uint end = min(envLightIndex, totalLightCount);
        if (begin >= end) return false;

        uint count = end - begin;
        lightID = begin + min((uint)(rand01(randomState) * count), count - 1);
        return true;
    }

    bool sampleLightByPdf(inout uint randomState, out uint lightID, out float2 sampleUV)
    {
        lightID = 0;
        sampleUV = rand2(randomState);
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
            sampleUV = rand2(randomState);
            return true;
        }

        uint cat = available[min((uint)(rand01(randomState) * n), n - 1)];

        if (cat == 0u)
        {
            if (sampleLocalLightFromPdfTexture(randomState, lightID))
            {
                sampleUV = rand2(randomState);
                return true;
            }
        }
        else if (cat == 1u)
        {
            if (sampleInfiniteAnalyticFromPmf(randomState, lightID))
            {
                sampleUV = rand2(randomState);
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
        sampleUV = rand2(randomState);
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

        uint state = initRandomSeed(uint2(index, gFrameIndex), gRandomSeed ^ 0x9E3779B9u);
        uint sampledLight;
        float2 sampledUV;
        sampleLightByPdf(state, sampledLight, sampledUV);
        gPresampledLightIndex[index] = sampledLight;
    }

    uint2 getPrevPixel(uint2 pixel)
    {
        float2 curr = float2(pixel) + 0.5f;
        float2 motion = mvec[pixel];
        float2 prev = curr + motion * gFrameDim;
        int2 prevPix = int2(floor(prev));

        if (prevPix.x < 0 || prevPix.y < 0 || prevPix.x >= gFrameDim.x || prevPix.y >= gFrameDim.y) return pixel;
        return uint2(prevPix);
    }

    bool areSurfacesCompatible(const SurfaceData a, const SurfaceData b)
    {
        if (!a.valid || !b.valid) return false;
        if (dot(a.normal, b.normal) < 0.95f) return false;

        float3 dP = b.position - a.position;
        if (abs(dot(dP, a.normal)) > 0.01f) return false;
        return true;
    }

    Reservoir combineReservoirs(uint2 pixel, SurfaceData surf, Reservoir r1, Reservoir r2, uint state)
    {
        Reservoir s = Reservoir();
        if (!surf.valid) return s;

        if (r1.M > 0 && r1.W > 0.f)
        {
            PolymorphicLightSample ls1;
            if (samplePolymorphicLight(surf, r1.lightID, r1.sampleUV, ls1))
            {
                float w1 = calcTargetPdf(surf, ls1.direction, ls1.Li) * (float)r1.M * r1.W;
                if (w1 > 0.f) s.update(r1.lightID, r1.sampleUV, w1, state);
            }
        }

        if (r2.M > 0 && r2.W > 0.f)
        {
            PolymorphicLightSample ls2;
            if (samplePolymorphicLight(surf, r2.lightID, r2.sampleUV, ls2))
            {
                float w2 = calcTargetPdf(surf, ls2.direction, ls2.Li) * (float)r2.M * r2.W;
                if (w2 > 0.f) s.update(r2.lightID, r2.sampleUV, w2, state);
            }
        }

        s.M = r1.M + r2.M;
        if (s.M == 0 || s.wsum <= 0.f) { s.W = 0.f; return s; }

        PolymorphicLightSample lsS;
        if (!samplePolymorphicLight(surf, s.lightID, s.sampleUV, lsS)) { s.W = 0.f; return s; }
        float phS = calcTargetPdf(surf, lsS.direction, lsS.Li);
        s.W = (phS > 0.f) ? min(s.wsum / ((float)s.M * phS), 10.f) : 0.f;
        return s;
    }

    Reservoir combineReservoirSet(uint2 pixel, SurfaceData surf, Reservoir rs[9], uint count, uint state)
    {
        Reservoir s = Reservoir();
        if (!surf.valid) return s;

        uint sumM = 0;
        for (uint i = 0; i < 9; ++i)
        {
            if (i >= count) break;
            Reservoir r = rs[i];
            sumM += r.M;
            if (r.M == 0 || r.W <= 0.f) continue;

            PolymorphicLightSample ls;
            if (!samplePolymorphicLight(surf, r.lightID, r.sampleUV, ls)) continue;
            float ph = calcTargetPdf(surf, ls.direction, ls.Li);
            s.update(r.lightID, r.sampleUV, r.W * ph * r.M, state);
        }

        s.M = sumM;
        if (s.M == 0) { s.W = 0.f; return s; }

        PolymorphicLightSample lsS;
        if (!samplePolymorphicLight(surf, s.lightID, s.sampleUV, lsS)) { s.W = 0.f; return s; }
        float ph = calcTargetPdf(surf, lsS.direction, lsS.Li);
        s.W = (ph > 0.f) ? min(s.wsum / (ph * (float)s.M), 10.f) : 0.f;
        return s;
    }

    void init(uint2 pixel)
    {
        if (any(pixel >= gFrameDim)) return;

        uint pixelIndex = pixel.y * gFrameDim.x + pixel.x;
        SurfaceData surf;
        bool validSurface = buildSurfaceData(pixel, surf);
        surfaceData[getSurfaceIndex(pixel, currentSurfaceBufferIndex)] = surf.pack();

        static const int kMaxSampleCount = 64;
        int n = clamp(gRIS_M, 0, kMaxSampleCount);
        uint lightCount = totalLightCount;
        uint randomState = initRandomSeed(pixel, gRandomSeed);

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

            uint p = 0;
            float2 uv = rand2(randomState);
            uint totalPresampled = max(gPresampledTileCount * gPresampledTileSize, 1u);
            if (gPresampledTileCount > 0 && gPresampledTileSize > 0)
            {
                uint pixelLinear = pixel.y * gFrameDim.x + pixel.x;
                uint tileId = (pixelLinear + (uint)i + gFrameIndex * 17u) % gPresampledTileCount;
                uint slot = min((uint)(rand01(randomState) * gPresampledTileSize), gPresampledTileSize - 1);
                uint entry = min(tileId * gPresampledTileSize + slot, totalPresampled - 1);
                p = min(gPresampledLightIndex[entry], lightCount - 1);

                if (p == envLightIndex)
                {
                    float2 envUv;
                    if (sampleEnvFromPdfTexture(randomState, envUv)) uv = envUv;
                }
            }
            else
            {
                sampleLightByPdf(randomState, p, uv);
            }

            if (p >= totalLightCount) continue;

            float wi = calcw(surf, p, uv);
            r.update(p, uv, wi, randomState);
        }

        PolymorphicLightSample ls;
        if (r.M > 0 && samplePolymorphicLight(surf, r.lightID, r.sampleUV, ls))
        {
            float ph = calcTargetPdf(surf, ls.direction, ls.Li);
            r.W = (ph > 0.f) ? min(r.wsum / (float(r.M) * ph), 10.f) : 0.f;
        }
        else r.W = 0.f;

        markExplosionStage(r, 1u);
        gReservoir[pixelIndex] = r;
    }

    void visibility(uint2 pixel)
    {
        if (any(pixel >= gFrameDim)) return;

        uint pixelIndex = pixel.y * gFrameDim.x + pixel.x;
        Reservoir r = gReservoir[pixelIndex];
        if (r.M == 0 || r.W <= 0.f) return;

        SurfaceData surf;
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
        bool visible = rayQuery.traceVisibilityRay(Ray(surf.position + surf.normal * 1e-3f, ls.direction, 0.f, tMax));
        if (!visible)
        {
            r.W = 0.f;
            r.wsum = 0.f;
        }

        markExplosionStage(r, 2u);
        gReservoir[pixelIndex] = r;
    }

    void temporalReuse(uint2 pixel)
    {
        if (any(pixel >= gFrameDim)) return;

        uint pixelIndex = pixel.y * gFrameDim.x + pixel.x;
        uint2 prevP = getPrevPixel(pixel);
        uint prevPixelIndex = prevP.y * gFrameDim.x + prevP.x;
        uint randomState = initRandomSeed(pixel, gRandomSeed);

        SurfaceData surf;
        SurfaceData prevSurf;
        loadSurface(pixel, currentSurfaceBufferIndex, true, surf);
        loadSurface(prevP, prevSurfaceBufferIndex, false, prevSurf);

        Reservoir baseR = gReservoir[pixelIndex];
        Reservoir prevR = gPrevReservoir[prevPixelIndex];
        //if (prevR.age >= kMaxReservoirAge) prevR = Reservoir();
        if (!areSurfacesCompatible(surf, prevSurf)) prevR = Reservoir();

        Reservoir outR = combineReservoirs(pixel, surf, baseR, prevR, randomState);
        if (outR.M > 0 && outR.W > 0.f) outR.age = min(prevR.age + 1u, kMaxReservoirAge);
        else outR.age = 0u;
        markExplosionStage(outR, 3u);
        gReservoir[pixelIndex] = outR;
    }

    void spatialReuse(uint2 pixel)
    {
        if (any(pixel >= gFrameDim)) return;

        uint pixelIndex = pixel.y * gFrameDim.x + pixel.x;
        uint randomState = initRandomSeed(pixel, gRandomSeed);

        SurfaceData surf;
        if (!loadSurface(pixel, currentSurfaceBufferIndex, true, surf))
        {
            gReservoir[pixelIndex] = Reservoir();
            return;
        }

        Reservoir rs[9];
        uint count = 0;
        uint inheritedAge = 0;

        for (int i = -1; i <= 1; ++i)
        {
            for (int j = -1; j <= 1; ++j)
            {
                int2 p = (int2)pixel + int2(i, j);
                if (p.x < 0 || p.y < 0 || p.x >= gFrameDim.x || p.y >= gFrameDim.y) continue;

                uint2 np = uint2(p);
                SurfaceData ns;
                if (!loadSurface(np, currentSurfaceBufferIndex, false, ns)) continue;
                if (!areSurfacesCompatible(surf, ns)) continue;

                uint index = np.y * gFrameDim.x + np.x;
                rs[count] = gPrevReservoir[index];
                //if (rs[count].age >= kMaxReservoirAge) continue;
                if (rs[count].age > inheritedAge) inheritedAge = rs[count].age;
                count++;
            }
        }

        Reservoir r = combineReservoirSet(pixel, surf, rs, count, randomState);
        if (r.M > 0 && r.W > 0.f) r.age = min(inheritedAge + 1u, kMaxReservoirAge);
        else r.age = 0u;
        markExplosionStage(r, 4u);
        gReservoir[pixelIndex] = r;
    }

    void shadeColor(uint2 pixel)
    {
        if (any(pixel >= gFrameDim)) return;

        uint pixelIndex = pixel.y * gFrameDim.x + pixel.x;
        Reservoir r = gReservoir[pixelIndex];
        SurfaceData surf;
        bool validSurface = loadSurface(pixel, currentSurfaceBufferIndex, true, surf);

        if (!validSurface || r.M == 0 || r.W <= 0.f)
        {
            color[pixel] = float4(0, 0, 0, 1);
            return;
        }

        if (gDebugShowWExplosion != 0)
        {
            color[pixel] = float4(getExplosionDebugColor(r.age, r.W), 1.f);
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
        bool visible = rayQuery.traceVisibilityRay(Ray(surf.position + surf.normal * 1e-3f, ls.direction, 0.f, tMax));
        if (!visible)
        {
            color[pixel] = float4(0, 0, 0, 1);
            return;
        }

        ShadingData sd;
        if (loadShadingData(pixel, gFrameDim, gScene.camera, vbuffer, sd))
        {
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
        }

        color[pixel] = float4(0, 0, 0, 1);
    }
};

ParameterBlock<RTDI> gRTDI;
