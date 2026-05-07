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
import FinReservoir;
import RandomHelper;
import Utils.Math.Ray;
#include "rtxdi/RtxdiMath.hlsli"
#define RTXDI_POLYMORPHIC_LIGHT_SAMPLING
#include "PolymorphicLight.slang"
#include "ResamplingFunctions.slangh"
#include "FinResamplingFunctions.slangh"
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
    uint biasCorrectionMode;
    float normalThreshold;
    float depthThreshold;
    float rayEpsilon;

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
    RWStructuredBuffer<FinReservoir> gReservoir;
    StructuredBuffer<FinReservoir> gPrevReservoir;

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

    bool useBasicBiasCorrection()
    {
        return biasCorrectionMode != 0u;
    }

    float scalarizeTargetPdf(float3 targetPdf)
    {
        return max(luminance(max(targetPdf, 0.f)), 0.f);
    }

    void discardReservoirSample(inout Reservoir r, uint channel)
    {
        if (channel >= 3 || channel < 0) return;
        r.clearLight(channel);
        r.uv[channel] = float2(0.f, 0.f);
        r.wsum[channel] = 0.f;
        r.targetPdf[channel] = 0.f;
    }

    void discardFinReservoirSample(inout FinReservoir r)
    {
        r.clearLight();
        r.uv = 0.f;
        r.wsum = 0.f;
        r.targetPdf = 0.f;
    }

    bool reservoirHasUnifiedSample(Reservoir r)
    {
        int referenceLight = -1;
        float2 referenceUv = 0.f;

        [unroll]
        for (uint channel = 0; channel < 3; ++channel)
        {
            if (!RTDI_IsValidReservoir(r, channel)) continue;

            int light = r.getLightIndex(channel);
            float2 uv = r.uv[channel];

            if (referenceLight < 0)
            {
                referenceLight = light;
                referenceUv = uv;
                continue;
            }

            if (light != referenceLight) return false;
            if (any(abs(uv - referenceUv) > 1e-6f)) return false;
        }

        return referenceLight >= 0;
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
        float3 resamplingNormal = props.guideNormal;
        if (dot(resamplingNormal, hit.V) <= 0.f)
        {
            // Avoid backfacing resampling normals turning the whole pixel into an
            // invalid reservoir at grazing angles on folds/creases.
            resamplingNormal = hit.getOrientedFaceNormal();
        }

        sd.position = hit.computeRayOrigin();
        sd.viewDepth = length(hit.posW - gScene.camera.data.posW);
        sd.normal = resamplingNormal;
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
    *   1 lights to 3 targetPdf
    */
    float3 calcTargetPdf(RTAB_Surface surf, float3 wi, float3 Li, uint2 pixel)
    {
        return calcSurfaceTargetPdf(surf, wi, Li);
    }

    float calcSurfaceTargetFactor(RTAB_Surface surf, float3 wi)
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

        float roughness = surf.roughness;
        float specular = surf.specular;
        float D = evalNdfGGX(roughness * roughness, NdotH);
        float G = evalMaskingSmithGGXSeparable(roughness, NdotL, NdotV);
        float F = specular < 1e-8f ? 0.f : (evalFresnelSchlick(specular, 1.f, LdotH) / max(specular, 1e-6f));

        float diffuseBRDF = NdotL * (1.f / PI);
        float specularBRDF = (D * G * F) / max(4.f * NdotV, 1e-6f);
        return max(0.f, lerp(specularBRDF, diffuseBRDF, surf.diffuseProb));
    }

    float3 calcSurfaceTargetPdf(RTAB_Surface surf, float3 wi, float3 Li)
    {
        float factor = calcSurfaceTargetFactor(surf, wi);
        return factor * max(Li, 0.f);
    }
    //3 lights to 3 targetpdf
    //TODO: ya suo this
    float3 calcTargetPdf(RTAB_Surface surf, float3 wi[3], float3 Li[3], uint2 pixel) {
        return calcSurfaceTargetPdf(surf, wi, Li);
    }

    float3 calcSurfaceTargetPdf(RTAB_Surface surf, float3 wi[3], float3 Li[3])
    {
        float3 targetPdf = 0.f;

        [unroll]
        for (uint channel = 0; channel < 3; ++channel)
        {
            if (all(wi[channel] == 0.f)) continue;
            targetPdf[channel] = calcSurfaceTargetFactor(surf, wi[channel]) * max(Li[channel][channel], 0.f);
        }

        return targetPdf;
    }

    float3 calcReservoirTargetPdf(RTAB_Surface surf, Reservoir reservoir, uint2 pixel)
    {
        float3 wi[3] = {};
        float3 Li[3] = {};
        bool hasValidChannel = false;

        [unroll]
        for (uint channel = 0; channel < 3; ++channel)
        {
            if (!RTDI_IsValidReservoir(reservoir, channel)) continue;

            PolymorphicLightSample ls;
            if (samplePolymorphicLight(surf, reservoir.getLightIndex(channel), reservoir.uv[channel], ls))
            {
                wi[channel] = ls.direction;
                Li[channel] = ls.Li;
                hasValidChannel = true;
            }
        }

        return hasValidChannel ? calcTargetPdf(surf, wi, Li, pixel) : float3(0.f);
    }

    float3 calcReservoirRgbTargetPdf(RTAB_Surface surf, Reservoir reservoir, uint2 pixel)
    {
        float3 wi[3] = {};
        float3 Li[3] = {};
        bool hasValidChannel = false;

        [unroll]
        for (uint channel = 0; channel < 3; ++channel)
        {
            if (!RTDI_IsValidReservoir(reservoir, channel)) continue;

            PolymorphicLightSample ls;
            if (samplePolymorphicLight(surf, reservoir.getLightIndex(channel), reservoir.uv[channel], false, ls))
            {
                wi[channel] = ls.direction;
                Li[channel] = ls.Li;
                hasValidChannel = true;
            }
        }

        return hasValidChannel ? calcTargetPdf(surf, wi, Li, pixel) : float3(0.f);
    }

    float3 calcReservoirSurfaceTargetPdf(RTAB_Surface surf, Reservoir reservoir)
    {
        float3 wi[3] = {};
        float3 Li[3] = {};
        bool hasValidChannel = false;

        [unroll]
        for (uint channel = 0; channel < 3; ++channel)
        {
            if (!RTDI_IsValidReservoir(reservoir, channel)) continue;

            PolymorphicLightSample ls;
            if (samplePolymorphicLight(surf, reservoir.getLightIndex(channel), reservoir.uv[channel], ls))
            {
                wi[channel] = ls.direction;
                Li[channel] = ls.Li;
                hasValidChannel = true;
            }
        }

        return hasValidChannel ? calcSurfaceTargetPdf(surf, wi, Li) : float3(0.f);
    }

    float3 calcReservoirRgbSurfaceTargetPdf(RTAB_Surface surf, Reservoir reservoir)
    {
        float3 wi[3] = {};
        float3 Li[3] = {};
        bool hasValidChannel = false;

        [unroll]
        for (uint channel = 0; channel < 3; ++channel)
        {
            if (!RTDI_IsValidReservoir(reservoir, channel)) continue;

            PolymorphicLightSample ls;
            if (samplePolymorphicLight(surf, reservoir.getLightIndex(channel), reservoir.uv[channel], false, ls))
            {
                wi[channel] = ls.direction;
                Li[channel] = ls.Li;
                hasValidChannel = true;
            }
        }

        return hasValidChannel ? calcSurfaceTargetPdf(surf, wi, Li) : float3(0.f);
    }

    float3 calcReservoirHelperPdf(RTAB_Surface surf, Reservoir reservoir)
    {
        float3 helperPdf = 0.f;

        [unroll]
        for (uint channel = 0; channel < 3; ++channel)
        {
            if (!RTDI_IsValidReservoir(reservoir, channel)) continue;

            PolymorphicLightSample ls;
            if (samplePolymorphicLight(surf, reservoir.getLightIndex(channel), reservoir.uv[channel], ls))
            {
                helperPdf[channel] = max(surfaceBrdfPdf(surf, ls.direction), 0.f);
            }
        }

        return helperPdf;
    }

    float3 calcFinReservoirTargetPdfVector(RTAB_Surface surf, FinReservoir reservoir, uint2 pixel)
    {
        if (!RTDI_IsValidFinReservoir(reservoir)) return 0.f;

        PolymorphicLightSample ls;
        if (!samplePolymorphicLight(surf, reservoir.getLightIndex(), reservoir.uv, ls))
            return 0.f;

        return calcTargetPdf(surf, ls.direction, ls.Li, pixel);
    }

    float3 calcFinReservoirRgbTargetPdf(RTAB_Surface surf, FinReservoir reservoir, uint2 pixel)
    {
        if (!RTDI_IsValidFinReservoir(reservoir)) return 0.f;

        PolymorphicLightSample ls;
        if (!samplePolymorphicLight(surf, reservoir.getLightIndex(), reservoir.uv, false, ls))
            return 0.f;

        return calcTargetPdf(surf, ls.direction, ls.Li, pixel);
    }

    float calcFinReservoirTargetPdf(RTAB_Surface surf, FinReservoir reservoir, uint2 pixel)
    {
        return scalarizeTargetPdf(calcFinReservoirTargetPdfVector(surf, reservoir, pixel));
    }

    float calcFinReservoirSurfaceTargetPdf(RTAB_Surface surf, FinReservoir reservoir)
    {
        if (!RTDI_IsValidFinReservoir(reservoir)) return 0.f;

        PolymorphicLightSample ls;
        if (!samplePolymorphicLight(surf, reservoir.getLightIndex(), reservoir.uv, ls))
            return 0.f;

        return scalarizeTargetPdf(calcSurfaceTargetPdf(surf, ls.direction, ls.Li));
    }

    float3 calcFinReservoirRgbSurfaceTargetPdf(RTAB_Surface surf, FinReservoir reservoir)
    {
        if (!RTDI_IsValidFinReservoir(reservoir)) return 0.f;

        PolymorphicLightSample ls;
        if (!samplePolymorphicLight(surf, reservoir.getLightIndex(), reservoir.uv, false, ls))
            return 0.f;

        return calcSurfaceTargetPdf(surf, ls.direction, ls.Li);
    }

    float calcFinReservoirHelperPdf(RTAB_Surface surf, FinReservoir reservoir)
    {
        if (!RTDI_IsValidFinReservoir(reservoir)) return 0.f;

        PolymorphicLightSample ls;
        if (!samplePolymorphicLight(surf, reservoir.getLightIndex(), reservoir.uv, ls))
            return 0.f;

        return max(surfaceBrdfPdf(surf, ls.direction), 0.f);
    }

    void finalizeBasicBiasCorrection(inout FinReservoir reservoir, float normalizationNumerator, float normalizationDenominator)
    {
        float denominator = reservoir.targetPdf * normalizationDenominator;
        bool invalid = denominator <= 0.f || normalizationNumerator <= 0.f;
        reservoir.wsum = invalid ? 0.f : (reservoir.wsum * normalizationNumerator) / max(denominator, 1e-6f);
    }

    void finalizeBasicBiasCorrection(inout Reservoir reservoir, float3 normalizationNumerator, float3 normalizationDenominator)
    {
        float3 denominator = reservoir.targetPdf * normalizationDenominator;
        bool3 invalid = (denominator <= 0.f) | (normalizationNumerator <= 0.f);
        reservoir.wsum = select(
            invalid,
            0.f,
            (reservoir.wsum * normalizationNumerator) / max(denominator, 1e-6f)
        );
    }

    bool3 streamFinReservoirToVectorReservoir(inout Reservoir targetR, FinReservoir sample, float random, float3 targetP)
    {
        bool3 result = false;
        float sampleNormalization = sample.M * sample.wsum;
        float3 risWeight = targetP * sampleNormalization;
        targetR.M += sample.M;
        targetR.wsum += risWeight;

        [unroll]
        for (uint channel = 0; channel < 3; ++channel)
        {
            if (random * targetR.wsum[channel] < risWeight[channel])
            {
                targetR.setLight(channel, sample.getLightIndex());
                targetR.uv[channel] = sample.uv;
                targetR.targetPdf[channel] = targetP[channel];
                result[channel] = true;
            }
        }

        return result;
    }

    FinReservoir compressVectorReservoirToFinReservoir(Reservoir source, inout RH_RandomSamplerState randomState)
    {
        FinReservoir result = FinReservoir();

        uint validChannelCount = 0u;
        [unroll]
        for (uint channel = 0; channel < 3; ++channel)
        {
            if (RTDI_IsValidReservoir(source, channel) && source.targetPdf[channel] > 0.f && source.wsum[channel] > 0.f)
                validChannelCount++;
        }

        if (validChannelCount == 0u) return result;

        float misWeight = 1.f / float(validChannelCount);

        [unroll]
        for (uint channel = 0; channel < 3; ++channel)
        {
            if (!RTDI_IsValidReservoir(source, channel)) continue;
            if (source.targetPdf[channel] <= 0.f || source.wsum[channel] <= 0.f) continue;

            RTDI_StreamFinReservoirSample(
                result,
                source.getLightIndex(channel),
                source.uv[channel],
                RH_GetNextRandom(randomState),
                source.targetPdf[channel],
                source.wsum[channel] * misWeight,
                1.f
            );
        }

        RTDI_FinalizeFinResampling(result, 1.f, max(result.M, 1.f));
        return result;
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

    float lightBrdfMisPdf(
        RTAB_Surface surface,
        PolymorphicLightSample lightSample,
        float lightSelectionPdf,
        float lightMisWeight,
        bool isEnvironmentMap,
        float brdfMisWeight)
    {
        float lightSolidAnglePdf = lightSample.pdf;
        if (brdfMisWeight <= 0.f ||
            lightSample.lightType == PolymorphicLightType::Analytic ||
            lightSolidAnglePdf <= 0.f ||
            isinf(lightSolidAnglePdf) ||
            isnan(lightSolidAnglePdf))
        {
            return lightMisWeight * lightSelectionPdf;
        }

        float brdfPdf = surfaceBrdfPdf(surface, lightSample.direction);
        float sourcePdfWrtSolidAngle = lightSelectionPdf * lightSolidAnglePdf;
        float blendedPdfWrtSolidAngle = lightMisWeight * sourcePdfWrtSolidAngle + brdfMisWeight * brdfPdf;
        return blendedPdfWrtSolidAngle / lightSolidAnglePdf;
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

        int topMip = int(mipCount) - 1;
        float averageValue = max(localLightPdfTexture.Load(int3(0, 0, topMip)).x, 0.f);
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

    Reservoir buildLocalReservoir(
        RTAB_Surface surf,
        uint sampleCount,
        float localMisWeight,
        float brdfMisWeight,
        float normalizationDenominator,
        inout RH_RandomSamplerState randomState,
        uint tileIndex,
        uint2 pixel)
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

            uint sampledLight = 0u;
            float sourcePdf = 0.f;
            if (gPresampledTileCount > 0u && gPresampledTileSize > 0u)
            {
                uint2 presampled = gPresampledLightIndex[entry];
                sampledLight = presampled.x;
                if (sampledLight >= firstInfiniteAnalyticLight) continue;
                float invSourcePdf = asfloat(presampled.y);
                sourcePdf = (invSourcePdf > 0.f) ? (1.f / invSourcePdf) : 0.f;
            }
            else if (sampleLocalLightFromPdfTexture(randomState, sampledLight))
            {
                sourcePdf = calcLocalLightSourcePdf(int(sampledLight));
            }
            else
            {
                continue;
            }
            lightID = int(sampledLight);

            float3 ph = 0.f;
            float3 w = 0.f;
            if (lightID >= 0)
            {
                PolymorphicLightSample ls;
                if (samplePolymorphicLight(surf, lightID, uv, ls))
                {
                    float p = lightBrdfMisPdf(surf, ls, sourcePdf, localMisWeight, false, brdfMisWeight);
                    if (p > 0.f)
                    {
                        ph = calcTargetPdf(surf, ls.direction, ls.Li, pixel);
                        w = ph / p;
                    }
                }
            }

            r.update(lightID, uv, w, ph, 1.f, randomState);
        }

        if (r.M > 0.f && any(r.targetPdf > 0.f))
        {
            float denom = max(normalizationDenominator, 1e-6f);
            r.wsum = r.wsum / max(r.targetPdf * denom, 1e-6f);
        }
        else
        {
            r.wsum = 0.f;
        }
        r.M = 1.f;
        return r;
    }

    Reservoir buildInfiniteReservoir(
        RTAB_Surface surf,
        uint sampleCount,
        inout RH_RandomSamplerState randomState,
        uint2 pixel
    )
    {
        Reservoir r = Reservoir();
        if (sampleCount == 0u) return r;

        [loop]
        for (uint i = 0; i < sampleCount; ++i)
        {
            uint sampledLight = 0u;
            if (!sampleInfiniteAnalyticFromPmf(randomState, sampledLight)) continue;

            float2 uv = RH_GetNextRandom2D(randomState);
            PolymorphicLightSample ls;
            if (!samplePolymorphicLight(surf, int(sampledLight), uv, ls)) continue;

            uint count = envLightIndex - firstInfiniteAnalyticLight;
            float sourcePdf = (count > 0u) ? (1.f / float(count)) : 0.f;
            float3 ph = calcTargetPdf(surf, ls.direction, ls.Li, pixel);
            if (sourcePdf <= 0.f || all(ph <= 0.f)) continue;

            r.update(int(sampledLight), uv, ph / sourcePdf, ph, 1.f, randomState);
        }

        RTDI_FinalizeResampling(r, 1.0f, max(r.M, 1.f));
        r.M = (sampleCount > 0u) ? 1.f : 0.f;
        return r;
    }

    Reservoir buildEnvironmentReservoir(
        RTAB_Surface surf,
        uint sampleCount,
        float envMisWeight,
        float brdfMisWeight,
        float normalizationDenominator,
        inout RH_RandomSamplerState randomState,
        uint tileIndex,
        uint2 pixel)
    {
        Reservoir r = Reservoir();
        if (sampleCount == 0u || envLightIndex >= totalLightCount) return r;

        [loop]
        for (uint i = 0; i < sampleCount; ++i)
        {
            float2 uv = float2(0.5f, 0.5f);
            float sourcePdf = 0.f;
            uint totalPresampled = max(gPresampledTileCount * gPresampledTileSize, 1u);
            uint slot = (gPresampledTileSize > 0u) ? min((uint)(RH_GetNextRandom(randomState) * gPresampledTileSize), gPresampledTileSize - 1u) : 0u;
            uint entry = min(tileIndex * max(gPresampledTileSize, 1u) + slot, totalPresampled - 1u);

            if (gPresampledTileCount > 0u && gPresampledTileSize > 0u)
            {
                uint2 envSample = gPresampledEnvData[entry];
                float invSourcePdf = asfloat(envSample.y);
                if (invSourcePdf <= 0.f) continue;
                uv = unpackUv16(envSample.x);
                sourcePdf = 1.f / invSourcePdf;
            }
            else if (sampleEnvFromPdfTexture(randomState, uv))
            {
                sourcePdf = calcEnvironmentMapSamplingPdf(gScene.envMap.uvToWorld(uv));
            }
            else
            {
                continue;
            }

            PolymorphicLightSample ls;
            if (!samplePolymorphicLight(surf, int(envLightIndex), uv, ls)) continue;

            float p = lightBrdfMisPdf(surf, ls, sourcePdf, envMisWeight, true, brdfMisWeight);
            if (p <= 0.f) continue;

            float3 ph = calcTargetPdf(surf, ls.direction, ls.Li, pixel);
            float3 w = ph / p;
            r.update(int(envLightIndex), uv, w, ph, 1.f, randomState);
        }

        if (r.M > 0.f && any(r.targetPdf > 0.f))
        {
            float denom = max(normalizationDenominator, 1e-6f);
            r.wsum = r.wsum / max(r.targetPdf * denom, 1e-6f);
        }
        else
        {
            r.wsum = 0.f;
        }
        r.M = 1.f;
        return r;
    }

    Reservoir buildBrdfReservoir(
        RTAB_Surface surf,
        uint sampleCount,
        float localMisWeight,
        float envMisWeight,
        float brdfMisWeight,
        float normalizationDenominator,
        inout RH_RandomSamplerState randomState,
        uint2 pixel)
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
                if (samplePolymorphicLight(surf, lightID, uv, candidateSample))
                {
                    lightSourcePdf = calcLocalLightSourcePdf(lightID);
                }
            }
            else if (!hitAnything && envLightIndex < totalLightCount)
            {
                isEnvSample = true;
                lightID = int(envLightIndex);
                uv = gScene.envMap.worldToUv(sampleDir);
                if (samplePolymorphicLight(surf, lightID, uv, candidateSample))
                {
                    lightSourcePdf = calcEnvironmentMapSamplingPdf(sampleDir);
                }
            }

            if (lightID < 0 || lightSourcePdf <= 0.f || !candidateSample.valid)
            {
                continue;
            }

            float3 targetPdf = calcTargetPdf(surf, candidateSample.direction, candidateSample.Li, pixel);
            float lightMisWeight = isEnvSample ? envMisWeight : localMisWeight;
            float blendedSourcePdf = lightMisWeight * lightSourcePdf + brdfMisWeight * brdfPdf;
            if (blendedSourcePdf <= 0.f || all(targetPdf <= 0.f))
            {
                continue;
            }

            float3 w = targetPdf / blendedSourcePdf;
            r.update(lightID, uv, w, targetPdf, 1.f, randomState);
        }

        if (r.M > 0.f && any(r.targetPdf > 0.f))
        {
            float denom = max(normalizationDenominator, 1e-6f);
            r.wsum = r.wsum / max(r.targetPdf * denom, 1e-6f);
        }
        else
        {
            r.wsum = 0.f;
        }
        r.M = 1.f;
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

    bool areMaterialsSimilar(const RTAB_Surface a, const RTAB_Surface b)
    {
        if (!a.valid || !b.valid) return false;

        const float diffuseThreshold = 0.2f;
        const float specularThreshold = 0.2f;
        const float roughnessThreshold = 0.15f;
        const float lobeMixThreshold = 0.2f;

        if (abs(a.diffuse - b.diffuse) > diffuseThreshold) return false;
        if (abs(a.specular - b.specular) > specularThreshold) return false;
        if (abs(a.roughness - b.roughness) > roughnessThreshold) return false;
        if (abs(a.diffuseProb - b.diffuseProb) > lobeMixThreshold) return false;

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

    FinReservoir convertReservoirToFinReservoir(RTAB_Surface surf, Reservoir source, inout RH_RandomSamplerState randomState, uint2 pixel)
    {
        FinReservoir result = FinReservoir();
        float proposalPdf[3];
        float sumPdf = 0;
        uint selChannel = 0;
        [unroll]
        for (uint channel = 0; channel < 3; ++channel) {
            proposalPdf[channel] = 1 / source.wsum[channel];
            sumPdf += proposalPdf[channel];
        }
        [unroll]
        for (uint channel = 0; channel < 3; ++channel)
        {
            if (!RTDI_IsValidReservoir(source, channel)) continue;

            PolymorphicLightSample ls;
            if (!samplePolymorphicLight(surf, source.getLightIndex(channel), source.uv[channel], ls))
                continue;

            float sampleTarget = scalarizeTargetPdf(calcTargetPdf(surf, ls.direction, ls.Li, pixel));
            if (sampleTarget <= 0.f) continue;

            bool sel = RTDI_StreamFinReservoirSample(
                result,
                source.getLightIndex(channel),
                source.uv[channel],
                RH_GetNextRandom(randomState),
                sampleTarget,
                source.wsum[channel],
                1.f
            );
            if (sel) selChannel = channel;
        }

        float normalizationDenominator = sumPdf;
        RTDI_FinalizeFinResampling(result, proposalPdf[selChannel], normalizationDenominator);
        return result;
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

        FinReservoir fin = FinReservoir();
        if (!validSurface || totalLightCount == 0)
        {
            gReservoir[pixelIndex] = fin;
            return;
        }

        uint numMisSamples = gLocalRIS_M + gEnvRIS_M + gBrdfRIS_M;
        float localMisWeight = (numMisSamples > 0u) ? (float(gLocalRIS_M) / float(numMisSamples)) : 0.f;
        float envMisWeight = (numMisSamples > 0u) ? (float(gEnvRIS_M) / float(numMisSamples)) : 0.f;
        float brdfMisWeight = (numMisSamples > 0u) ? (float(gBrdfRIS_M) / float(numMisSamples)) : 0.f;
        uint localTileIndex = (gPresampledTileCount > 0u) ? min((uint)(RH_GetNextRandom(coherentState) * gPresampledTileCount), gPresampledTileCount - 1u) : 0u;
        uint envTileIndex = (gPresampledTileCount > 0u) ? min((uint)(RH_GetNextRandom(coherentState) * gPresampledTileCount), gPresampledTileCount - 1u) : 0u;

        Reservoir localReservoir = buildLocalReservoir(surf, gLocalRIS_M, localMisWeight, brdfMisWeight, float(max(numMisSamples, 1u)), randomState, localTileIndex, pixel);
        Reservoir infiniteReservoir = buildInfiniteReservoir(surf, gInfiniteRIS_M, randomState, pixel);
        Reservoir envReservoir = buildEnvironmentReservoir(surf, gEnvRIS_M, envMisWeight, brdfMisWeight, float(max(numMisSamples, 1u)), randomState, envTileIndex, pixel);
        Reservoir brdfReservoir = buildBrdfReservoir(surf, gBrdfRIS_M, localMisWeight, envMisWeight, brdfMisWeight, float(max(numMisSamples, 1u)), randomState, pixel);

        Reservoir initReservoir = Reservoir();
        RTDI_CombineReservoirs(initReservoir, localReservoir, 0.5f, localReservoir.targetPdf);
        RTDI_CombineReservoirs(initReservoir, infiniteReservoir, RH_GetNextRandom(randomState), infiniteReservoir.targetPdf);
        RTDI_CombineReservoirs(initReservoir, envReservoir, RH_GetNextRandom(randomState), envReservoir.targetPdf);
        RTDI_CombineReservoirs(initReservoir, brdfReservoir, RH_GetNextRandom(randomState), brdfReservoir.targetPdf);
        RTDI_FinalizeResampling(initReservoir, 1.0f, 1.0f);
        initReservoir.M = (localReservoir.M > 0.f || infiniteReservoir.M > 0.f || envReservoir.M > 0.f || brdfReservoir.M > 0.f) ? 1.f : 0.f;

        fin = convertReservoirToFinReservoir(surf, initReservoir, randomState, pixel);
        gReservoir[pixelIndex] = fin;
    }

    void visibility(uint2 pixel)
    {
        if (any(pixel >= gFrameDim)) return;

        uint pixelIndex = pixel.y * gFrameDim.x + pixel.x;
        FinReservoir r = gReservoir[pixelIndex];

        RTAB_Surface surf;
        if (!loadSurface(pixel, currentSurfaceBufferIndex, true, surf))
        {
            //gReservoir[pixelIndex] = Reservoir();
            return;
        }
        if (RTDI_IsValidFinReservoir(r))
        {
            PolymorphicLightSample ls;
            if (!samplePolymorphicLight(surf, r.getLightIndex(), r.uv, ls))
            {
                discardFinReservoirSample(r);
                gReservoir[pixelIndex] = r;
                return;
            }

            float tMax = (ls.distance >= FLT_MAX * 0.5f) ? FLT_MAX : max(ls.distance - rayEpsilon, 0.f);
            SceneRayQuery<0> rayQuery;
            bool visible = rayQuery.traceVisibilityRay(Ray(surf.position, ls.direction, rayEpsilon, tMax), RAY_FLAG_CULL_NON_OPAQUE);
            if (!visible)
            {
                discardFinReservoirSample(r);
            }
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
            gReservoir[pixelIndex] = FinReservoir();
            return;
        }
        FinReservoir baseR = gReservoir[pixelIndex];

        FinReservoir state = FinReservoir();
        RTDI_CombineFinReservoirs(state, baseR, 0.5f, baseR.targetPdf);
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
            if (!areMaterialsSimilar(surf, tempSurface)) continue;

            findNeighbor = true;
            spatialOffset = idx - prevPixel;
            prevPixel = idx;
            break;
        }

        bool selectedPrevSample = false;
        if (findNeighbor) {
            uint prevPixelIndex = uint(prevPixel.y) * gFrameDim.x + uint(prevPixel.x);
            FinReservoir prevSample = gPrevReservoir[prevPixelIndex];
            prevSample.M = min(prevSample.M, historyLimit);
            float currentPdf = calcFinReservoirTargetPdf(surf, prevSample, pixel);
            selectedPrevSample = RTDI_CombineFinReservoirs(state, prevSample, RH_GetNextRandom(randomState), currentPdf);
        }

        if (useBasicBiasCorrection())
        {
            float pi = state.targetPdf;
            float piSum = state.targetPdf * baseR.M;

            if (findNeighbor)
            {
                uint normPrevPixelIndex = uint(prevPixel.y) * gFrameDim.x + uint(prevPixel.x);
                FinReservoir normPrevSample = gPrevReservoir[normPrevPixelIndex];
                normPrevSample.M = min(normPrevSample.M, historyLimit);
                float temporalP = calcFinReservoirSurfaceTargetPdf(tempSurface, state);
                pi = selectedPrevSample ? temporalP : pi;
                piSum += temporalP * normPrevSample.M;
            }

            finalizeBasicBiasCorrection(state, pi, piSum);
        }
        else
        {
            RTDI_FinalizeFinResampling(state, 1.0f, state.M);
        }

        gReservoir[pixelIndex] = state;
    }

    void spatiotemporalReuse(uint2 pixel, uint maxHistoryLength, uint numSamples, float temporalSamplingRadius, float spatialSamplingRadius)
    {
        if (any(pixel >= gFrameDim)) return;

        uint pixelIndex = pixel.y * gFrameDim.x + pixel.x;
        RH_RandomSamplerState rng = RH_InitRandomState(pixel, gFrameIndex, 5);

        RTAB_Surface surf;
        if (!loadSurface(pixel, currentSurfaceBufferIndex, true, surf))
        {
            gReservoir[pixelIndex] = FinReservoir();
            return;
        }

        FinReservoir baseR = gReservoir[pixelIndex];
        FinReservoir state = FinReservoir();
        RTDI_CombineFinReservoirs(state, baseR, 0.5f, baseR.targetPdf);

        uint historyLimit = min(RTDI_ReservoirMaxM, maxHistoryLength * baseR.M);

        int2 reprojectedPixel = getPrevPixel(pixel, rng);
        float3 motion = float3(mvec[pixel], 0.f); // TODO handle linear z
        float expectedPrevLinearDepth = RTAB_GetSurfaceLinearDepth(surf) + motion.z;

        int2 matchedTemporalPixel = int2(0, 0);
        bool foundTemporalSurface = false;
        RTAB_Surface temporalSurface;

        for (int i = 0; i < 9; ++i)
        {
            int2 offset = int2(0, 0);
            if (i > 0)
            {
                offset.x = int((RH_GetNextRandom(rng) - 0.5f) * temporalSamplingRadius);
                offset.y = int((RH_GetNextRandom(rng) - 0.5f) * temporalSamplingRadius);
            }

            int2 idx = reprojectedPixel + offset;
            if (!loadSurface(idx, prevSurfaceBufferIndex, true, temporalSurface)) continue;

            if (!isValidNeighbor(
                surf.normal, temporalSurface.normal,
                expectedPrevLinearDepth, RTAB_GetSurfaceLinearDepth(temporalSurface),
                normalThreshold, depthThreshold))
                continue;
            if (!areMaterialsSimilar(surf, temporalSurface)) continue;

            matchedTemporalPixel = idx;
            foundTemporalSurface = true;
            break;
        }

        bool temporalSelected = false;
        if (foundTemporalSurface)
        {
            uint temporalPixelIndex = uint(matchedTemporalPixel.y) * gFrameDim.x + uint(matchedTemporalPixel.x);
            FinReservoir prevSample = gPrevReservoir[temporalPixelIndex];
            prevSample.M = min(prevSample.M, historyLimit);
            float currentPdf = calcFinReservoirTargetPdf(surf, prevSample, pixel);
            temporalSelected = RTDI_CombineFinReservoirs(state, prevSample, RH_GetNextRandom(rng), currentPdf);
        }

        uint numSpatialSamples = min(numSamples, 31u);
        uint startIdx = uint(RH_GetNextRandom(rng) * float(gNeighborOffsetMask));
        RTAB_Surface neighborSurface;
        uint cachedSpatialResult = 0u;
        int selectedSpatialNeighbor = -1;

        for (uint i = 0; i < numSpatialSamples; ++i)
        {
            uint sampleIdx = (startIdx + i) & gNeighborOffsetMask;
            int2 spatialOffset = int2(float2(gNeighborOffsetBuffer[sampleIdx].xy) * spatialSamplingRadius);
            int2 idx = reprojectedPixel + spatialOffset;

            if (foundTemporalSurface && all(idx == matchedTemporalPixel))
                continue;

            if (!loadSurface(idx, prevSurfaceBufferIndex, true, neighborSurface)) continue;

            if (!isValidNeighbor(
                surf.normal, neighborSurface.normal,
                RTAB_GetSurfaceLinearDepth(surf), RTAB_GetSurfaceLinearDepth(neighborSurface),
                normalThreshold, depthThreshold))
                continue;
            if (!areMaterialsSimilar(surf, neighborSurface)) continue;

            uint neighborIndex = uint(idx.y) * gFrameDim.x + uint(idx.x);
            FinReservoir neighborSample = gPrevReservoir[neighborIndex];
            neighborSample.M = min(neighborSample.M, historyLimit);
            float neighborTargetPdf = calcFinReservoirTargetPdf(surf, neighborSample, pixel);

            cachedSpatialResult |= (1u << i);
            bool selectedNeighbor = RTDI_CombineFinReservoirs(state, neighborSample, RH_GetNextRandom(rng), neighborTargetPdf);
            if (selectedNeighbor) selectedSpatialNeighbor = int(i);
        }

        if (useBasicBiasCorrection())
        {
            float pi = state.targetPdf;
            float piSum = state.targetPdf * baseR.M;

            if (foundTemporalSurface)
            {
                uint normTemporalPixelIndex = uint(matchedTemporalPixel.y) * gFrameDim.x + uint(matchedTemporalPixel.x);
                FinReservoir normPrevSample = gPrevReservoir[normTemporalPixelIndex];
                normPrevSample.M = min(normPrevSample.M, historyLimit);
                float temporalP = calcFinReservoirSurfaceTargetPdf(temporalSurface, state);
                pi = temporalSelected ? temporalP : pi;
                piSum += temporalP * normPrevSample.M;
            }

            for (uint i = 0; i < numSpatialSamples; ++i)
            {
                if ((cachedSpatialResult & (1u << i)) == 0u) continue;

                uint sampleIdx = (startIdx + i) & gNeighborOffsetMask;
                int2 spatialOffset = int2(float2(gNeighborOffsetBuffer[sampleIdx].xy) * spatialSamplingRadius);
                int2 idx = reprojectedPixel + spatialOffset;

                RTAB_Surface cachedNeighborSurface;
                if (!loadSurface(idx, prevSurfaceBufferIndex, true, cachedNeighborSurface)) continue;

                uint neighborIndex = uint(idx.y) * gFrameDim.x + uint(idx.x);
                FinReservoir normNeighborSample = gPrevReservoir[neighborIndex];
                normNeighborSample.M = min(normNeighborSample.M, historyLimit);
                float ps = calcFinReservoirSurfaceTargetPdf(cachedNeighborSurface, state);
                if (selectedSpatialNeighbor == int(i)) pi = ps;
                piSum += ps * normNeighborSample.M;
            }

            finalizeBasicBiasCorrection(state, pi, piSum);
        }
        else
        {
            RTDI_FinalizeFinResampling(state, 1.0f, state.M);
        }

        gReservoir[pixelIndex] = state;
    }

    /** spatial resampling
    */
    void spatialReuse(uint2 pixel, uint numSamples, float samplingRadius)
    {
        if (any(pixel >= gFrameDim)) return;
        uint centerPixelIndex = pixel.y * gFrameDim.x + pixel.x;
        RH_RandomSamplerState rng = RH_InitRandomState(pixel, gFrameIndex, 3);
        FinReservoir centerSample = gPrevReservoir[centerPixelIndex];
        RTAB_Surface centerSurface;
        if (!loadSurface(pixel, currentSurfaceBufferIndex, true, centerSurface))
        {
            gReservoir[centerPixelIndex] = FinReservoir();
            return;
        }

        FinReservoir state = FinReservoir();
        RTDI_CombineFinReservoirs(state, centerSample, 0.5f, centerSample.targetPdf);

        // TODO When the length of history is less than targetHistoryLength
        // customize the number of samples to select.
        uint numSpatialSamples = min(numSamples, 32u);
        int i = 0;
        RTAB_Surface neighborSurface;
        uint startIdx = uint(RH_GetNextRandom(rng) * (float)gNeighborOffsetMask);
        uint cachedSpatialResult = 0u;
        int selectedNeighbor = -1;
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
            if (!areMaterialsSimilar(centerSurface, neighborSurface))
                continue;
            uint neighborIndex = idx.y * gFrameDim.x + idx.x;
            FinReservoir neighborSample = gPrevReservoir[neighborIndex];
            float neighborTargetPdf = calcFinReservoirTargetPdf(centerSurface, neighborSample, pixel);
            if (neighborTargetPdf <= 0.f) continue;
            cachedSpatialResult |= (1u << i);
            bool selectedThisNeighbor = RTDI_CombineFinReservoirs(state, neighborSample, RH_GetNextRandom(rng), neighborTargetPdf);
            if (selectedThisNeighbor) selectedNeighbor = i;
        }
        if (useBasicBiasCorrection())
        {
            float pi = state.targetPdf;
            float piSum = state.targetPdf * centerSample.M;

            for (uint i = 0; i < numSpatialSamples; ++i)
            {
                if ((cachedSpatialResult & (1u << i)) == 0u) continue;

                uint sampleIdx = (startIdx + i) & gNeighborOffsetMask;
                int2 spatialOffset = int2(float2(gNeighborOffsetBuffer[sampleIdx].xy) * samplingRadius);
                int2 idx = int2(pixel) + spatialOffset;
                RTAB_Surface cachedNeighborSurface;
                if (!loadSurface(idx, currentSurfaceBufferIndex, true, cachedNeighborSurface)) continue;

                uint neighborIndex = idx.y * gFrameDim.x + idx.x;
                FinReservoir normNeighborSample = gPrevReservoir[neighborIndex];
                float ps = calcFinReservoirTargetPdf(cachedNeighborSurface, state, uint2(idx));
                if (selectedNeighbor == int(i)) pi = ps;
                piSum += ps * normNeighborSample.M;
            }

            finalizeBasicBiasCorrection(state, pi, piSum);
        }
        else
        {
            RTDI_FinalizeFinResampling(state, 1.0f, state.M);
        }

        gReservoir[centerPixelIndex] = state;
    }

    void shadeColor(uint2 pixel)
    {
        if (any(pixel >= gFrameDim)) return;
        uint pixelIndex = pixel.y * gFrameDim.x + pixel.x;
        FinReservoir r = gReservoir[pixelIndex];
        ShadingData sd;
        if (loadShadingData(pixel, gFrameDim, gScene.camera, vbuffer, sd))
        {

            RTAB_Surface surf;
            bool validSurface = loadSurface(pixel, currentSurfaceBufferIndex, true, surf);
            let lod = ExplicitLodTextureSampler(0.0f);
            let hints = getMaterialInstanceHints();
            let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
            float3 illumination = {};
            if (!validSurface)
            {
                color[pixel] = float4(0,0,0,1);
                return;
            }
            if (RTDI_IsValidFinReservoir(r))
            {
                PolymorphicLightSample ls;
                if (samplePolymorphicLight(surf, r.getLightIndex(), r.uv, ls))
                {
                    float tMax = (ls.distance >= FLT_MAX * 0.5f) ? FLT_MAX : max(ls.distance - rayEpsilon, 0.f);
                    SceneRayQuery<1> rayQuery;
                    const Ray ray = Ray(sd.computeRayOrigin(), ls.direction, 0.f, tMax);
                    if (rayQuery.traceVisibilityRay(ray))
                    {
                        TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(pixel, gFrameIndex);
                        sd.mtl.setActiveLobes((uint)LobeType::DiffuseReflection);
                        float3 diffuseBsdf = mi.eval(sd, ls.direction, sg);
                        sd.mtl.setActiveLobes((uint)LobeType::SpecularReflection);
                        float3 specularBsdf = mi.eval(sd, ls.direction, sg);
                        illumination = (diffuseBsdf + specularBsdf) * ls.Li * r.wsum;
                    }
                }
            }
            let props = mi.getProperties(sd);
            color[pixel] = float4(illumination + props.emission, 1);
            return;
        } else {
            float3 dir = gScene.camera.computeRayPinhole(pixel, gFrameDim).dir;
            color[pixel] = float4(gScene.envMap.eval(dir),1);
        }
    }
};

ParameterBlock<RTDI> gRTDI;
