import Utils.Math.HashUtils;
import Utils.Math.ShadingFrame;
import Utils.Geometry.IntersectionHelpers;
import Utils.Debug.PixelDebug;
import Utils.Color.ColorMap;
import Utils.Color.ColorHelpers;
import Utils.Sampling.SampleGenerator;

import Scene.RaytracingInline;
import Scene.Scene;

import Utils.Math.Ray;
#define PI 3.14159265358979323846
cbuffer CB{
    uint gRIS_M;
    float2 gFrameDim;
    uint gFrameIndex;
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
    float3 posW;
    float3 nrmW;
    float3 woW;     //指向相机的方向
    float3 albedo;  //反射率
    float  rough;   //粗糙度
    uint   valid;   //像素是否命中几何
};
//蓄水池
RWStructuredBuffer<Reservoir> gReservoir;
StructuredBuffer<Reservoir> gPrevReservoir;

//Alias Table
StructuredBuffer<float> gAliasProb;
StructuredBuffer<uint> gAliasIndex;
//输入纹理
Texture2D<PackedHitInfo> vbuffer;

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

uint initRandomSeed(uint2 pixel, uint frameIndex)
{
    uint seed = 0x9e3779b9u;
    seed ^= hash32(pixel.x + 0x68bc21ebu);
    seed ^= hash32(pixel.y + 0x02e5be93u);
    seed ^= hash32(frameIndex + 0x967a889bu);
    return max(hash32(seed), 1u);
}
// =========================================================================================
// 从Vbuffer中解压出surf信息
// =========================================================================================
Surface DecodeSurfaceFromHit(const float3 orig, const float3 dir, const HitInfo hit)
{
    Surface s = {};
    s.valid = 0;

    // 1) 从 hit 得到 instance/material + 顶点属性 v
    uint materialID = 0;
    VertexData v = {};

    switch (hit.getType())
    {
    case HitType::Triangle:
    {
        const TriangleHit th = hit.getTriangleHit();
        materialID = gScene.getMaterialID(th.instanceID);
        v = gScene.getVertexData(th);
        break;
    }
    case HitType::DisplacedTriangle:
    {
        const DisplacedTriangleHit dh = hit.getDisplacedTriangleHit();
        materialID = gScene.getMaterialID(dh.instanceID);
        v = gScene.getVertexData(dh, -dir);
        break;
    }
    case HitType::Curve:
    {
        const CurveHit ch = hit.getCurveHit();
        materialID = gScene.getMaterialID(ch.instanceID);
        v = gScene.getVertexDataFromCurve(ch);
        break;
    }
    case HitType::SDFGrid:
    {
        const SDFGridHit sh = hit.getSDFGridHit();
        materialID = gScene.getMaterialID(sh.instanceID);
        v = gScene.getVertexDataFromSDFGrid(sh, orig, dir);
        break;
    }
    default:
        return s;
    }

    // 2) 生成 ShadingData
    const ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -dir);

    // 3) 取材质属性（albedo/roughness 等）
    let lod = ExplicitLodTextureSampler(0.f);
    let mi = gScene.materials.getMaterialInstance(sd, lod);
    let props = mi.getProperties(sd);

    // 4) 填充 Surface
    s.posW = sd.posW;

    s.nrmW = sd.frame.N;
    s.woW = sd.V;
    s.albedo = props.diffuseReflectionAlbedo;
    s.rough = props.roughness;

    s.valid = 1;
    return s;
}
// ==================================
// 计算w的各项
// ==================================

//BRDF
float3 CalcBRDF(Surface surf) {
    return surf.albedo / PI;
}
// 几何项
float CalcG(Surface surf, EmissiveTriangle tri, float2 bary) {
    EmissiveTriangle light = tri;
    float b0 = 1 - bary.x - bary.y;
    float3 samp = light.posW[0] * b0 + light.posW[1] * bary.x + light.posW[2] * bary.y; // rebuild samp
    float3 L = samp - surf.posW;
    float dist2 = dot(L, L);
    float3 wi = normalize(L);
    float cosThetaS = saturate(dot(surf.nrmW, wi));
    float cosThetaL = saturate(dot(light.normal, -wi));
    return (cosThetaS * cosThetaL / dist2);
}

// Le
float3 CalcLe(EmissiveTriangle tri, float2 bary) {
    return gScene.materials.evalEmissive(tri.materialID, tri.getTexCoord(float3(1 - bary.x - bary.y, bary.x, bary.y)));
}
// w
float Calcw(Surface surf, uint triLightID, float2 bary) {
    EmissiveTriangle tri = gScene.lightCollection.getTriangle(triLightID);
    uint count = gScene.lightCollection.getTriangleCount();
    float p = gAliasProb[triLightID] / (float)count;
    p = p / tri.area;
    float ph = luminance(CalcBRDF(surf) * CalcLe(tri, bary));
    ph = ph * CalcG(surf, tri, bary);
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
    uint randomState = initRandomSeed(pixel, gFrameIndex);
    uint pixelIndex = pixel.y * (uint)gFrameDim.x + pixel.x;
    if (lightCount == 0 || n == 0) {
        gReservoir[pixelIndex] = Reservoir();
        return;
    }
    //get surface
    HitInfo hit = unpackHitInfo(vbuffer[pixel]);
    const Ray ray = gScene.camera.computeRayPinhole(pixel, (uint2)gFrameDim);
    Surface surf = DecodeSurfaceFromHit(ray.origin, ray.dir, hit);
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
                // samp point on triangle
                float2 u = rand2(randomState);
                float su0 = sqrt(u.x);
                float s1 = su0 * (1.0 - u.y);
                float s2 = su0 * u.y;
                float2 bary = float2(s1, s2);
                float wi = Calcw(surf, p, bary);
                r.update(p, bary, wi, randomState);
            }
        }
        // calc W
        EmissiveTriangle tri = gScene.lightCollection.getTriangle(r.lightID);
        float ph = luminance(CalcBRDF(surf) * CalcLe(tri, r.bary));
        ph = ph * CalcG(surf, tri, r.bary);
        if (r.M > 0 && ph > 0.f) {
            r.W = r.wsum / (float)r.M / ph;
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

    uint pixelIndex = pixel.y * (uint)gFrameDim.x + pixel.x;
    Reservoir r = gReservoir[pixelIndex];
    if (r.M == 0 || r.W <= 0.f) {
        return;
    }

    HitInfo hit = unpackHitInfo(vbuffer[pixel]);
    const Ray cameraRay = gScene.camera.computeRayPinhole(pixel, (uint2)gFrameDim);
    Surface surf = DecodeSurfaceFromHit(cameraRay.origin, cameraRay.dir, hit);
    if (surf.valid == 0) {
        gReservoir[pixelIndex] = Reservoir();
        return;
    }

    EmissiveTriangle tri = gScene.lightCollection.getTriangle(r.lightID);
    float3 lightPosW = EvalLightSamplePosW(tri, r.bary);
    float3 toLight = lightPosW - surf.posW;
    float dist = length(toLight);
    if (dist <= 1e-4f) {
        r.W = 0.f;
        r.wsum = 0.f;
        gReservoir[pixelIndex] = r;
        return;
    }

    float3 dir = toLight / dist;
    float3 origin = surf.posW + surf.nrmW * 1e-3f;
    float tMax = max(dist - 2e-3f, 0.f);

    SceneRayQuery<1> rayQuery;
    bool visible = (tMax > 0.f) && rayQuery.traceVisibilityRay(Ray(origin, dir, 0.f, tMax));
    if (!visible) {
        r.W = 0.f;
        r.wsum = 0.f;
        gReservoir[pixelIndex] = r;
    }
}
