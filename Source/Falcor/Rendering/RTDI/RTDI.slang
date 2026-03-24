import Utils.Math.HashUtils;
import Utils.Math.ShadingFrame;
import Utils.Geometry.IntersectionHelpers;
import Utils.Debug.PixelDebug;
import Utils.Color.ColorMap;
import Utils.Color.ColorHelpers;
import Utils.Sampling.SampleGenerator;
import Utils.Sampling.TinyUniformSampleGenerator;

import Scene.RaytracingInline;
import Scene.Scene;
import RenderPasses.ReSTIR_DI.LoadShadingData;

import Utils.Math.Ray;
#include "RandomHelper.slangh"

#define PI 3.14159265358979323846

struct Reservoir
{
    uint lightID = 0;
    float2 bary = float2(0, 0); // (b1, b2), b0 = 1 - b1 - b2

    float W = 0.0;
    float wsum = 0.0;
    uint M = 0;
    uint age = 0;

    [mutating] void update(uint light, float2 b, float w, inout uint state)
    {
        wsum = wsum + w;
        M = M + 1;
        if (rand01(state) < (w / wsum))
        {
            lightID = light;
            bary = b;
        }
    }
};

struct Surface
{
    ShadingData sd;
    float3 nrmW;
    uint valid;
};

struct RTDI
{
    // Runtime parameters.
    uint gRIS_M;
    uint2 gFrameDim;
    uint gFrameIndex;
    uint gRandomSeed;
    uint gDebugShowWExplosion;
    float gDebugWThreshold;

    // Internal algorithm resources.
    RWStructuredBuffer<Reservoir> gReservoir;
    StructuredBuffer<Reservoir> gPrevReservoir;

    StructuredBuffer<float> gAliasProb;
    StructuredBuffer<uint> gAliasIndex;
    StructuredBuffer<float> gLightPmf;

    Texture2D<PackedHitInfo> vbuffer;
    Texture2D<float2> mvec;
    Texture2D envLightLuminanceTexture;
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

    Surface decodeSurfaceFromHit(uint2 pixel)
    {
        Surface s = {};
        s.valid = 0;
        ShadingData sd;
        if (!loadShadingData(pixel, gFrameDim, gScene.camera, vbuffer, sd))
            return s;

        const uint hints = getMaterialInstanceHints();
        let lod = ExplicitLodTextureSampler(0.f);
        let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
        let props = mi.getProperties(sd);

        s.sd = sd;
        s.nrmW = props.guideNormal;
        s.valid = 1;
        return s;
    }

    float3 evalLightSamplePosW(EmissiveTriangle tri, float2 bary)
    {
        float b0 = 1.f - bary.x - bary.y;
        return tri.posW[0] * b0 + tri.posW[1] * bary.x + tri.posW[2] * bary.y;
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
    float calcTargetPdf(Surface surf, float3 wi, uint2 pixel, uint triID) {
        if (surf.valid == 0)
            return 0.f;

        const uint hints = getMaterialInstanceHints();
        let lod = ExplicitLodTextureSampler(0.f);
        let mi = gScene.materials.getMaterialInstance(surf.sd, lod, hints);
        let props = mi.getProperties(surf.sd);

        float3 N = normalize(surf.nrmW);
        float3 V = normalize(surf.sd.V);
        float3 L = normalize(wi);
        float3 H = normalize(V + L);

        float NdotV = saturate(dot(N, V));
        float NdotL = saturate(dot(N, L));
        float NdotH = saturate(dot(N, H));
        float LdotH = saturate(dot(L, H));
        if (NdotL <= 0.f || NdotV <= 0.f)
            return 0.f;

        float roughness = max(props.roughness, 0.02f);
        float alpha = roughness * roughness;
        float D = evalNdfGGXLocal(alpha * alpha, NdotH);
        float G = evalMaskingSchlickGGX(roughness, NdotL, NdotV);

        float diffuseL = max(luminance(props.diffuseReflectionAlbedo), 0.f);
        float specularL = max(luminance(props.specularReflectance), 0.f);
        float F0 = saturate(specularL);
        float F = fresnelSchlickScalar(F0, LdotH);

        float diffuseProb = diffuseL / max(diffuseL + specularL, 1e-6f);
        float diffuseBRDF = diffuseL * NdotL * (1.f / PI);
        float specularBRDF = (D * G * F) / max(4.f * NdotL * NdotV, 1e-6f);
        float fullBRDF = max(0.f, lerp(specularBRDF, diffuseBRDF, diffuseProb));
        float3 Li = gScene.lightCollection.getAverageRadiance(triID);
        return fullBRDF * Li.x;
    }

    // float calcG(Surface surf, EmissiveTriangle tri, float2 bary)
    // {
    //     float3 samp = evalLightSamplePosW(tri, bary);
    //     float3 L = samp - surf.sd.posW;
    //     float dist2 = dot(L, L);
    //     float3 wi = normalize(L);
    //     float cosThetaS = saturate(dot(surf.nrmW, wi));
    //     float cosThetaL = saturate(dot(tri.normal, -wi));
    //     float g = (cosThetaS * cosThetaL / max(dist2, 0.01f));
    //     return min(g, 10.0f);
    // }

    float3 calcLe(EmissiveTriangle tri, float2 bary)
    {
        return gScene.materials.evalEmissive(tri.materialID, tri.getTexCoord(float3(1 - bary.x - bary.y, bary.x, bary.y)));
    }

    float calcw(Surface surf, uint triLightID, float2 bary, uint2 pixel)
    {
        EmissiveTriangle tri = gScene.lightCollection.getTriangle(triLightID);
        if (tri.area <= 0.f)
            return 0.f;

        float pmf = gLightPmf[triLightID];
        if (pmf <= 0.f)
            return 0.f;

        float3 lightPos = evalLightSamplePosW(tri, bary);
        float3 wi = normalize(lightPos - surf.sd.posW);

        float p = pmf / tri.area;
        if (p <= 0.f)
            return 0.f;

        float ph = calcTargetPdf(surf,wi,pixel,triLightID);
        return ph / p;
    }

    uint2 getPrevPixel(uint2 pixel)
    {
        float2 curr = float2(pixel) + 0.5f;
        float2 motion = mvec[pixel];
        float2 prev = curr + motion * gFrameDim;
        int2 prevPix = int2(floor(prev));

        if (prevPix.x < 0 || prevPix.y < 0 || prevPix.x >= gFrameDim.x || prevPix.y >= gFrameDim.y)
            return pixel;

        return uint2(prevPix);
    }

    bool areSurfacesCompatible(const Surface a, const Surface b)
    {
        if (a.valid == 0 || b.valid == 0)
            return false;

        const float nDot = dot(a.nrmW, b.nrmW);
        if (nDot < 0.95f)
            return false;

        float3 dP = b.sd.posW - a.sd.posW;
        const float planeDist = abs(dot(dP, a.nrmW));
        if (planeDist > 0.01f)
            return false;

        return true;
    }

    Reservoir combineReservoirs(uint2 pixel, const Surface surf, Reservoir r1, Reservoir r2, uint state)
    {
        Reservoir s = Reservoir();

        if (surf.valid == 0)
            return s;

        if (r1.M > 0 && r1.W > 0.f)
        {
            EmissiveTriangle tri1 = gScene.lightCollection.getTriangle(r1.lightID);
            float3 lightPos1 = evalLightSamplePosW(tri1, r1.bary);
            float3 wi1 = normalize(lightPos1 - surf.sd.posW);
            float ph1 = calcTargetPdf(surf, wi1, pixel, r1.lightID);
            float w1 = ph1 * (float)r1.M * r1.W;
            if (w1 > 0.f)
                s.update(r1.lightID, r1.bary, w1, state);
        }

        if (r2.M > 0 && r2.W > 0.f)
        {
            EmissiveTriangle tri2 = gScene.lightCollection.getTriangle(r2.lightID);
            float3 lightPos2 = evalLightSamplePosW(tri2, r2.bary);
            float3 wi2 = normalize(lightPos2 - surf.sd.posW);
            float ph2 = calcTargetPdf(surf, wi2, pixel, r2.lightID);
            float w2 = ph2 * (float)r2.M * r2.W;
            if (w2 > 0.f)
                s.update(r2.lightID, r2.bary, w2, state);
        }

        s.M = r1.M + r2.M;
        if (s.M == 0 || s.wsum <= 0.f)
        {
            s.W = 0.f;
            return s;
        }

        EmissiveTriangle triS = gScene.lightCollection.getTriangle(s.lightID);
        float3 lightPosS = evalLightSamplePosW(triS, s.bary);
        float3 wiS = normalize(lightPosS - surf.sd.posW);
        float phS = calcTargetPdf(surf, wiS, pixel, s.lightID);

        if (phS > 0.f)
        {
            s.W = s.wsum / ((float)s.M * phS);
            s.W = min(s.W, 10.0f);
        }
        else
        {
            s.W = 0.f;
        }

        return s;
    }

    Reservoir combineReservoirSet(uint2 pixel, const Surface surf, Reservoir rs[9], uint count, uint state)
    {
        Reservoir s = Reservoir();
        if (surf.valid == 0)
            return s;

        uint sumM = 0;
        for (uint i = 0; i < 9; i++)
        {
            if (i >= count)
                break;

            Reservoir r = rs[i];
            sumM = sumM + r.M;
            if (r.M == 0 || r.W <= 0.f)
                continue;

            EmissiveTriangle tri = gScene.lightCollection.getTriangle(r.lightID);
            float3 lightPos = evalLightSamplePosW(tri, r.bary);
            float3 wi = normalize(lightPos - surf.sd.posW);
            float ph = calcTargetPdf(surf, wi, pixel, r.lightID);
            s.update(r.lightID, r.bary, r.W * ph * r.M, state);
        }

        s.M = sumM;
        if (s.M == 0)
        {
            s.W = 0.f;
            return s;
        }

        EmissiveTriangle tri = gScene.lightCollection.getTriangle(s.lightID);
        float3 lightPos = evalLightSamplePosW(tri, s.bary);
        float3 wi = normalize(lightPos - surf.sd.posW);
        float ph = calcTargetPdf(surf, wi, pixel, s.lightID);

        if (ph <= 0.f)
        {
            s.W = 0.f;
            return s;
        }

        s.W = s.wsum / (ph * (float)s.M);
        s.W = min(s.W, 10.0f);
        return s;
    }

    void init(uint2 pixel)
    {
        if (pixel.x >= gFrameDim.x || pixel.y >= gFrameDim.y)
            return;

        static const int kMaxSampleCount = 64;
        int n = clamp(gRIS_M, 0, kMaxSampleCount);
        uint lightCount = gScene.lightCollection.getTriangleCount();
        uint randomState = initRandomSeed(pixel, gRandomSeed);
        uint pixelIndex = pixel.y * gFrameDim.x + pixel.x;

        if (lightCount == 0 || n == 0)
        {
            gReservoir[pixelIndex] = Reservoir();
            return;
        }

        Surface surf = decodeSurfaceFromHit(pixel);
        Reservoir r = Reservoir();

        if (surf.valid != 0)
        {
            [unroll]
            for (int i = 0; i < kMaxSampleCount; ++i)
            {
                if (i < n)
                {
                    uint p = (uint)(rand01(randomState) * lightCount);
                    if (rand01(randomState) > gAliasProb[p])
                        p = gAliasIndex[p];

                    if (gLightPmf[p] <= 0.f)
                        continue;

                    float2 u = rand2(randomState);
                    float su0 = sqrt(u.x);
                    float s1 = su0 * (1.0 - u.y);
                    float s2 = su0 * u.y;
                    float2 bary = float2(s1, s2);
                    float wi = calcw(surf, p, bary, pixel);
                    r.update(p, bary, wi, randomState);
                }
            }

            EmissiveTriangle tri = gScene.lightCollection.getTriangle(r.lightID);
            float3 lightPos = evalLightSamplePosW(tri, r.bary);
            float3 wi = normalize(lightPos - surf.sd.posW);
            float ph = calcTargetPdf(surf, wi, pixel, r.lightID);

            if (r.M > 0 && ph > 0.f)
            {
                r.W = r.wsum / (float)r.M / ph;
                r.W = min(r.W, 10.0f);
            }
            else
            {
                r.W = 0.f;
            }
        }

        markExplosionStage(r, 1u);
        gReservoir[pixelIndex] = r;
    }

    void visibility(uint2 pixel)
    {
        if (pixel.x >= gFrameDim.x || pixel.y >= gFrameDim.y)
            return;

        uint pixelIndex = pixel.y * gFrameDim.x + pixel.x;
        Reservoir r = gReservoir[pixelIndex];
        if (r.M == 0 || r.W <= 0.f)
            return;

        Surface surf = decodeSurfaceFromHit(pixel);
        if (surf.valid == 0)
        {
            gReservoir[pixelIndex] = Reservoir();
            return;
        }

        EmissiveTriangle tri = gScene.lightCollection.getTriangle(r.lightID);
        float3 lightPosW = evalLightSamplePosW(tri, r.bary);
        float3 toLight = lightPosW - surf.sd.posW;
        float dist = length(toLight);
        if (dist <= 1e-4f)
        {
            r.W = 0.f;
            r.wsum = 0.f;
            gReservoir[pixelIndex] = r;
            return;
        }

        float3 dir = toLight / dist;
        float3 origin = surf.sd.computeRayOrigin();
        float tMax = max(dist - 2e-3f, 0.f);

        SceneRayQuery<1> rayQuery;
        bool visible = (tMax > 0.f) && rayQuery.traceVisibilityRay(Ray(origin, dir, 0.f, tMax));
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
        if (pixel.x >= gFrameDim.x || pixel.y >= gFrameDim.y)
            return;

        uint pixelIndex = pixel.y * gFrameDim.x + pixel.x;
        uint2 prevP = getPrevPixel(pixel);
        uint prevPixelIndex = prevP.y * gFrameDim.x + prevP.x;
        uint randomState = initRandomSeed(pixel, gRandomSeed);

        Surface surf = decodeSurfaceFromHit(pixel);
        Surface prevSurf = decodeSurfaceFromHit(prevP);
        Reservoir baseR = gReservoir[pixelIndex];
        Reservoir prevR = gPrevReservoir[prevPixelIndex];

        if (!areSurfacesCompatible(surf, prevSurf))
            prevR = Reservoir();

        Reservoir outR = combineReservoirs(pixel, surf, baseR, prevR, randomState);
        outR.age = (baseR.age != 0) ? baseR.age : prevR.age;
        markExplosionStage(outR, 3u);
        gReservoir[pixelIndex] = outR;
    }

    void spatialReuse(uint2 pixel)
    {
        if (pixel.x >= gFrameDim.x || pixel.y >= gFrameDim.y)
            return;

        uint pixelIndex = pixel.y * gFrameDim.x + pixel.x;
        uint randomState = initRandomSeed(pixel, gRandomSeed);
        Surface surf = decodeSurfaceFromHit(pixel);

        if (surf.valid == 0)
        {
            gReservoir[pixelIndex] = Reservoir();
            return;
        }

        Reservoir rs[9];
        uint count = 0;
        uint inheritedAge = 0;
        for (int i = -1; i <= 1; i++)
        {
            for (int j = -1; j <= 1; j++)
            {
                int2 p = (int2)pixel + int2(i, j);
                if (p.x < 0 || p.x >= gFrameDim.x || p.y < 0 || p.y >= gFrameDim.y)
                    continue;

                uint2 np = uint2(p);
                Surface ns = decodeSurfaceFromHit(np);
                if (!areSurfacesCompatible(surf, ns))
                    continue;

                uint index = np.y * gFrameDim.x + np.x;
                rs[count] = gPrevReservoir[index];
                if (inheritedAge == 0 && rs[count].age != 0)
                    inheritedAge = rs[count].age;
                count++;
            }
        }

        Reservoir r = combineReservoirSet(pixel, surf, rs, count, randomState);
        r.age = inheritedAge;
        markExplosionStage(r, 4u);
        gReservoir[pixelIndex] = r;
    }
    void shadeColor(uint2 pixel)
    {
        if (pixel.x >= gFrameDim.x || pixel.y >= gFrameDim.y)
            return;

        uint pixelIndex = pixel.y * (uint)gFrameDim.x + pixel.x;
        Reservoir r = gReservoir[pixelIndex];
        Surface surf = decodeSurfaceFromHit(pixel);

        if (surf.valid == 0 || r.M == 0 || r.W <= 0)
        {
            color[pixel] = float4(0, 0, 0, 1);
            return;
        }

        if (gDebugShowWExplosion != 0)
        {
            color[pixel] = float4(getExplosionDebugColor(r.age, r.W), 1.f);
            return;
        }

        EmissiveTriangle tri = gScene.lightCollection.getTriangle(r.lightID);
        float3 lightPos = evalLightSamplePosW(tri, r.bary);
        float3 toLight = lightPos - surf.sd.posW;
        float dist = length(toLight);
        if (dist <= 1e-4f)
        {
            color[pixel] = float4(0, 0, 0, 1);
            return;
        }

        float3 wi = toLight / dist;
        SceneRayQuery<1> rayQuery;
        bool visible = rayQuery.traceVisibilityRay(Ray(surf.sd.computeRayOrigin(), wi, 0.f, max(dist - 2e-3f, 0.f)));
        if (!visible)
        {
            let lod = ExplicitLodTextureSampler(0.0f);
            let hints = getMaterialInstanceHints();
            let mi = gScene.materials.getMaterialInstance(surf.sd, lod, hints);
            let props = mi.getProperties(surf.sd);
            color[pixel] = float4(props.emission, 1);
            return;
        }
        ShadingData sd;
        if (loadShadingData(pixel, gRTDI.gFrameDim, gScene.camera, gRTDI.vbuffer, sd)) {
            float3 diffuse = {};
            float3 specular = {};
            let lod = ExplicitLodTextureSampler(0.0f);
            let hints = getMaterialInstanceHints();
            let mi = gScene.materials.getMaterialInstance(surf.sd, lod, hints);
            float3 Li = calcLe(tri, r.bary);
            TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(pixel, gRTDI.gFrameIndex);
            sd.mtl.setActiveLobes((uint)LobeType::DiffuseReflection);
            diffuse = mi.eval(sd, wi, sg) * Li;
            sd.mtl.setActiveLobes((uint)LobeType::SpecularReflection);
            specular = mi.eval(sd, wi, sg) * Li;
            let props = mi.getProperties(surf.sd);
            float3 Lo = diffuse + specular + props.emission;
            color[pixel] = float4(Lo, 1);
        }

    }
};

ParameterBlock<RTDI> gRTDI;







