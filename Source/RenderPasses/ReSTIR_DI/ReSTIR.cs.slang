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
import LoadShadingData;

import Utils.Math.Ray;
#define PI 3.14159265358979323846
cbuffer CB{
    uint gRIS_M;
    uint2 gFrameDim;
    uint gFrameIndex;
    uint gRandomSeed;
}
struct Reservoir
{
    uint lightID = 0;
    float2 bary = float2(0,0); //(b1,b2) , b0 = 1 - b1 - b2;

    float  W = 0.0;       // ps: not w
    float  wsum = 0.0;    //sum of weights
    uint M = 0;         // num of sample so far
    [mutating] void update(uint light, float2 b, float w, inout uint state) {
        wsum = wsum + w;
        M = M + 1;
        if (rand01(state) < (w / wsum)) {
            lightID = light;
            bary = b;
        }
    }
};
struct Surface
{
    ShadingData sd;
    float3 nrmW;
    uint   valid;   //像素是否命中几何
};
//蓄水池
RWStructuredBuffer<Reservoir> gReservoir;
StructuredBuffer<Reservoir> gPrevReservoir;

//Alias Table
StructuredBuffer<float> gAliasProb;
StructuredBuffer<uint> gAliasIndex;
StructuredBuffer<float> gLightPmf;
//输入纹理
Texture2D<PackedHitInfo> vbuffer;
Texture2D<float2> mvec;

// Output color
RWTexture2D<float4> color;

// ===============
// 随机数生成
// ===============
uint pcg32(inout uint state)
{
    uint prev = state;
    state = prev * 747796405u + 2891336453u;
    uint word = ((prev >> ((prev >> 28u) + 4u)) ^ prev) * 277803737u;
    return (word >> 22u) ^ word;
}

float rand01(inout uint state)
{
    // [0,1)
    return (pcg32(state) >> 8) * (1.0 / 16777216.0);
}

float2 rand2(inout uint state)
{
    return float2(rand01(state), rand01(state));
}

// Small integer hash mixer for RNG seed initialization.
uint hash32(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

uint initRandomSeed(uint2 pixel, uint gSeed)
{
    uint seed = 0x9e3779b9u;
    seed ^= hash32(pixel.x + 0x68bc21ebu);
    seed ^= hash32(pixel.y + 0x02e5be93u);
    seed ^= hash32(gSeed + 0x967a889bu);
    return max(hash32(seed), 1u);
}
// =========================================================================================
// 从Vbuffer中解压出surf信息（不依赖主射线重建）
// =========================================================================================
Surface DecodeSurfaceFromHit(uint2 pixel)
{
    Surface s = {};
    s.valid = 0;

    // Use the same shading data path as other render passes to ensure consistency.
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
// ==================================
// 计算w的各项
// ==================================

//BRDF (Diffuse + Specular GGX)
float3 CalcBRDF(Surface surf, float3 wi, uint2 pixel) {
    if (surf.valid == 0)
        return float3(0, 0, 0);
    const uint hints = getMaterialInstanceHints();
    let lod = ExplicitLodTextureSampler(0.f);
    let mi = gScene.materials.getMaterialInstance(surf.sd, lod, hints);
    uint sampleNum = gRandomSeed ^ gFrameIndex;
    TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(pixel, sampleNum);
    surf.sd.mtl.setActiveLobes((uint)LobeType::DiffuseReflection);
    let diffuse = mi.eval(surf.sd, wi, sg);
    surf.sd.mtl.setActiveLobes((uint)LobeType::SpecularReflection);
    let specular = mi.eval(surf.sd, wi, sg);
    return diffuse + specular;
}
// 几何项
float CalcG(Surface surf, EmissiveTriangle tri, float2 bary) {
    EmissiveTriangle light = tri;
    float b0 = 1 - bary.x - bary.y;
    float3 samp = light.posW[0] * b0 + light.posW[1] * bary.x + light.posW[2] * bary.y; // rebuild samp
    float3 L = samp - surf.sd.posW;
    float dist2 = dot(L, L);
    float3 wi = normalize(L);
    float cosThetaS = saturate(dot(surf.nrmW, wi));
    float cosThetaL = saturate(dot(light.normal, -wi));
    float g = (cosThetaS * cosThetaL / max(dist2, 0.01f));
    return min(g, 10.0f);
}

// Le
float3 CalcLe(EmissiveTriangle tri, float2 bary) {
    return gScene.materials.evalEmissive(tri.materialID, tri.getTexCoord(float3(1 - bary.x - bary.y, bary.x, bary.y)));
}
// w
float Calcw(Surface surf, uint triLightID, float2 bary, uint2 pixel) {
    EmissiveTriangle tri = gScene.lightCollection.getTriangle(triLightID);
    if (tri.area <= 0.f) return 0.f;

    float pmf = gLightPmf[triLightID];
    if (pmf <= 0.f) return 0.f;

    float3 lightPos = EvalLightSamplePosW(tri, bary);
    float3 wi = normalize(lightPos - surf.sd.posW);

    float p = pmf / tri.area;
    if (p <= 0.f) return 0.f;

    float3 brdf = CalcBRDF(surf, wi, pixel);
    float3 contrib = brdf * CalcLe(tri, bary) * CalcG(surf, tri, bary);

    float ph = luminance(contrib);
    ph = max(ph, 1e-6f);

    return ph / p;
}

float3 EvalLightSamplePosW(EmissiveTriangle tri, float2 bary)
{
    float b0 = 1.f - bary.x - bary.y;
    return tri.posW[0] * b0 + tri.posW[1] * bary.x + tri.posW[2] * bary.y;
}
// ====================================
// Stage 1
// Generate initial candidates
// ====================================
[numthreads(16,16,1)]
void ReSTIR_Init(uint3 dispatchThreadId : SV_DispatchThreadID) {
    static const int kMaxSampleCount = 64;
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= gFrameDim.x || pixel.y >= gFrameDim.y) {
        return;
    }
    int n = clamp(gRIS_M, 0, kMaxSampleCount);
    uint lightCount = gScene.lightCollection.getTriangleCount();
    uint randomState = initRandomSeed(pixel, gRandomSeed);
    uint pixelIndex = pixel.y * gFrameDim.x + pixel.x;
    if (lightCount == 0 || n == 0) {
        gReservoir[pixelIndex] = Reservoir();
        return;
    }
    //get surface
    Surface surf = DecodeSurfaceFromHit(pixel);
    // ===========================
    // RIS
    // ===========================
    Reservoir r = Reservoir();
    if (surf.valid != 0) {
        [unroll]
        for (int i = 0; i < kMaxSampleCount; ++i) {
            if (i < n) {
                // samp triangle
                uint p = (uint)(rand01(randomState) * lightCount);
                if (rand01(randomState) > gAliasProb[p]) {
                    p = gAliasIndex[p];
                }
                if (gLightPmf[p] <= 0.f) {
                    continue;
                }
                // samp point on triangle
                float2 u = rand2(randomState);
                float su0 = sqrt(u.x);
                float s1 = su0 * (1.0 - u.y);
                float s2 = su0 * u.y;
                float2 bary = float2(s1, s2);
                float wi = Calcw(surf, p, bary, pixel);
                r.update(p, bary, wi, randomState);
            }
        }
        // calc W
        EmissiveTriangle tri = gScene.lightCollection.getTriangle(r.lightID);
        float3 lightPos = EvalLightSamplePosW(tri, r.bary);
        float3 wi = normalize(lightPos - surf.sd.posW);
        float3 brdf = CalcBRDF(surf, wi, pixel);
        float3 contrib = brdf * CalcLe(tri, r.bary) * CalcG(surf, tri, r.bary);
        float ph = luminance(contrib);
        ph = max(ph, 1e-6f);
        if (r.M > 0 && ph > 0.f) {
            r.W = r.wsum / (float)r.M / ph;
            r.W = min(r.W, 10.0f);
        } else {
            r.W = 0.f;
        }
    }
    gReservoir[pixelIndex] = r;
}
// ============================
// stage 2
// Evaluate visibility for initial candidates
// ============================
[numthreads(16,16,1)]
void ReSTIR_Visibility(uint3 dispatchThreadId: SV_DispatchThreadID) {
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= gFrameDim.x || pixel.y >= gFrameDim.y) {
        return;
    }

    uint pixelIndex = pixel.y * gFrameDim.x + pixel.x;
    Reservoir r = gReservoir[pixelIndex];
    if (r.M == 0 || r.W <= 0.f) {
        return;
    }

    Surface surf = DecodeSurfaceFromHit(pixel);
    if (surf.valid == 0) {
        gReservoir[pixelIndex] = Reservoir();
        return;
    }

    EmissiveTriangle tri = gScene.lightCollection.getTriangle(r.lightID);
    float3 lightPosW = EvalLightSamplePosW(tri, r.bary);
    float3 toLight = lightPosW - surf.sd.posW;
    float dist = length(toLight);
    if (dist <= 1e-4f) {
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
    if (!visible) {
        r.W = 0.f;
        r.wsum = 0.f;
        gReservoir[pixelIndex] = r;
    }
}
uint2 getPrevPixel(uint2 pixel) {
    float2 curr = float2(pixel) + 0.5f;
    float2 motion = mvec[pixel];
    float2 prev = curr + motion * gFrameDim;
    int2 prevPix = int2(floor(prev));

    if (prevPix.x < 0 || prevPix.y < 0 || prevPix.x >= gFrameDim.x || prevPix.y >= gFrameDim.y) {
        return pixel;
    }
    return uint2(prevPix);
}

bool areSurfacesCompatible(const Surface a, const Surface b)
{
    if (a.valid == 0 || b.valid == 0) return false;

    const float nDot = dot(a.nrmW, b.nrmW);
    if (nDot < 0.95f) return false;

    float3 dP = b.sd.posW - a.sd.posW;
    const float planeDist = abs(dot(dP, a.nrmW));
    if (planeDist > 0.01f) return false;

    return true;
}

Reservoir CombineReservoirs(uint2 pixel, const Surface surf, Reservoir r1, Reservoir r2, uint state) {
    Reservoir s = Reservoir();

    if (surf.valid == 0) {
        return s;
    }

    if (r1.M > 0 && r1.W > 0.f)
    {
        EmissiveTriangle tri1 = gScene.lightCollection.getTriangle(r1.lightID);
        float3 lightPos1 = EvalLightSamplePosW(tri1, r1.bary);
        float3 wi1 = normalize(lightPos1 - surf.sd.posW);
        float3 brdf1 = CalcBRDF(surf, wi1, pixel);
        float3 contrib1 = brdf1 * CalcLe(tri1, r1.bary) * CalcG(surf, tri1, r1.bary);
        float ph1 = luminance(contrib1);
        ph1 = max(ph1, 1e-6f);
        float w1 = ph1 * (float)r1.M * r1.W;
        if (w1 > 0.f) s.update(r1.lightID, r1.bary, w1, state);
    }

    if (r2.M > 0 && r2.W > 0.f)
    {
        EmissiveTriangle tri2 = gScene.lightCollection.getTriangle(r2.lightID);
        float3 lightPos2 = EvalLightSamplePosW(tri2, r2.bary);
        float3 wi2 = normalize(lightPos2 - surf.sd.posW);
        float3 brdf2 = CalcBRDF(surf, wi2, pixel);
        float3 contrib2 = brdf2 * CalcLe(tri2, r2.bary) * CalcG(surf, tri2, r2.bary);
        float ph2 = luminance(contrib2);
        ph2 = max(ph2, 1e-6f);
        float w2 = ph2 * (float)r2.M * r2.W;
        if (w2 > 0.f) s.update(r2.lightID, r2.bary, w2, state);
    }

    s.M = r1.M + r2.M;
    if (s.M == 0 || s.wsum <= 0.f) {
        s.W = 0.f;
        return s;
    }

    EmissiveTriangle triS = gScene.lightCollection.getTriangle(s.lightID);
    float3 lightPosS = EvalLightSamplePosW(triS, s.bary);
    float3 wiS = normalize(lightPosS - surf.sd.posW);
    float3 brdfS = CalcBRDF(surf, wiS, pixel);
    float3 contribS = brdfS * CalcLe(triS, s.bary) * CalcG(surf, triS, s.bary);
    float phS = luminance(contribS);
    phS = max(phS, 1e-6f);
    if (phS > 0.f) {
        s.W = s.wsum / ((float)s.M * phS);
        s.W = min(s.W, 10.0f);
    } else {
        s.W = 0.f;
    }

    return s;
}
// =================
// stage 3
// =================
[numthreads(16, 16, 1)]
void ReSTIR_TemporalReuse(uint3 dispatchThreadId: SV_DispatchThreadID) {
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= gFrameDim.x || pixel.y >= gFrameDim.y) {
        return;
    }
    uint pixelIndex = pixel.y * gFrameDim.x + pixel.x;
    // get prev pixel
    uint2 prevP = getPrevPixel(pixel);
    uint prevPixelIndex = prevP.y * gFrameDim.x + prevP.x;
    // random seed
    uint randomState = initRandomSeed(pixel, gRandomSeed);
    // update
    Surface surf = DecodeSurfaceFromHit(pixel);
    Surface prevSurf = DecodeSurfaceFromHit(prevP);
    Reservoir prevR = gPrevReservoir[prevPixelIndex];
    if (!areSurfacesCompatible(surf, prevSurf))
    {
        prevR = Reservoir();
    }

    gReservoir[pixelIndex] = CombineReservoirs(pixel, surf, gReservoir[pixelIndex], prevR, randomState);
}
Reservoir combineReservoirSet(uint2 pixel, const Surface surf, Reservoir rs[9], uint count, uint state) {
    Reservoir s = Reservoir();
    if (surf.valid == 0) return s;
    uint sumM = 0;
    for (uint i = 0; i < 9; i++) {
        if (i >= count) break;
        Reservoir r = rs[i];
        sumM = sumM + r.M;
        if (r.M == 0 || r.W <= 0.f) continue;
        EmissiveTriangle tri = gScene.lightCollection.getTriangle(r.lightID);
        float3 lightPos = EvalLightSamplePosW(tri, r.bary);
        float3 wi = normalize(lightPos - surf.sd.posW);
        float3 brdf = CalcBRDF(surf, wi,pixel);
        float3 contrib = brdf * CalcLe(tri, r.bary) * CalcG(surf, tri, r.bary);
        float ph = luminance(contrib);
        ph = max(ph, 1e-6f);
        s.update(r.lightID, r.bary, r.W * ph * r.M, state);
    }
    s.M = sumM;
    if (s.M == 0) {
        s.W = 0.f;
        return s;
    }
    EmissiveTriangle tri = gScene.lightCollection.getTriangle(s.lightID);
    float3 lightPos = EvalLightSamplePosW(tri, s.bary);
    float3 wi = normalize(lightPos - surf.sd.posW);
    float3 brdf = CalcBRDF(surf, wi, pixel);
    float3 contrib = brdf * CalcLe(tri, s.bary) * CalcG(surf, tri, s.bary);
    float ph = luminance(contrib);
    ph = max(ph, 1e-6f);
    if (ph <= 0.f) {
        s.W = 0.f;
        return s;
    }
    s.W = s.wsum / (ph * (float)s.M);
    s.W = min(s.W, 10.0f);
    return s;
}
// ===================
// stage 4
// ===================
[numthreads(16, 16, 1)]
void ReSTIR_SpatialReuse(uint3 dispatchThreadId: SV_DispatchThreadID) {
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= gFrameDim.x || pixel.y >= gFrameDim.y) return;
    uint pixelIndex = pixel.y * gFrameDim.x + pixel.x;
    uint randomState = initRandomSeed(pixel, gRandomSeed);
    Surface surf = DecodeSurfaceFromHit(pixel);
    if (surf.valid == 0) {
        gReservoir[pixelIndex] = Reservoir();
        return;
    }
    Reservoir rs[9];
    uint count = 0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int2 p = (int2)pixel;
            p = p + int2(i, j);
            if (p.x < 0 || p.x >= gFrameDim.x || p.y < 0 || p.y >= gFrameDim.y) {
                continue;
            }
            uint2 np = uint2(p);
            Surface ns = DecodeSurfaceFromHit(np);
            if (!areSurfacesCompatible(surf, ns)) {
                continue;
            }
            uint index = np.y * gFrameDim.x + np.x;
            rs[count] = gPrevReservoir[index];
            count++;
        }
    }
    Reservoir r = combineReservoirSet(pixel, surf, rs, count, randomState);
    gReservoir[pixelIndex] = r;
}
// ========================
// Stage 5
// shade pixel
// ========================
[numthreads(16,16,1)]
void ReSTIR_ShadeColor(uint3 dispatchThreadId: SV_DispatchThreadID) {
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= gFrameDim.x || pixel.y >= gFrameDim.y) return;
    uint pixelIndex = pixel.y * (uint)gFrameDim.x + pixel.x;
    Reservoir r = gReservoir[pixelIndex];
    Surface surf = DecodeSurfaceFromHit(pixel);
    if (surf.valid == 0 || r.M == 0 || r.W <= 0) {
        color[pixel] = float4(0, 0, 0, 1);
        return;
    }
    EmissiveTriangle tri = gScene.lightCollection.getTriangle(r.lightID);
    float3 lightPos = EvalLightSamplePosW(tri, r.bary);
    float3 toLight = lightPos - surf.sd.posW;
    float dist = length(toLight);
    if (dist <= 1e-4f) {
        color[pixel] = float4(0, 0, 0, 1);
        return;
    }

    float3 wi = toLight / dist;
    SceneRayQuery<1> rayQuery;
    bool visible = rayQuery.traceVisibilityRay(Ray(surf.sd.computeRayOrigin(), wi, 0.f, max(dist - 2e-3f, 0.f)));
    if (!visible) {
        let lod = ExplicitLodTextureSampler(0.0f);
        let hints = getMaterialInstanceHints();
        let mi = gScene.materials.getMaterialInstance(surf.sd, lod, hints);
        let props = mi.getProperties(surf.sd);
        color[pixel] = float4(props.emission, 1);
        return;
    }

    float3 f = CalcBRDF(surf, wi, pixel) * CalcLe(tri, r.bary);
    float g = CalcG(surf, tri, r.bary);
    let lod = ExplicitLodTextureSampler(0.0f);
    let hints = getMaterialInstanceHints();
    let mi = gScene.materials.getMaterialInstance(surf.sd, lod, hints);
    let props = mi.getProperties(surf.sd);
    float3 Lo = f * g * r.W + props.emission;
    color[pixel] = float4(Lo, 1);
}
