import RandomHelper;
struct Reservoir
{
    int3 lightID = int3(-1, -1, -1);
    float2 uv[3] = {};
    float3 wsum = 0.0;
    float3 targetPdf = 0.0;
    float M = 0.0;

    [mutating] void update(int light, float2 uv, float3 w, float3 ph, float m, inout RH_RandomSamplerState state)
    {
        wsum = wsum + w;
        M = M + m;
        [unroll]
        for (uint i = 0; i < 3; i++) {
            if (RH_GetNextRandom(state) < (w[i] / max(wsum[i], 1e-6f)))
            {
                lightID[i] = light;
                this.uv[i] = uv;
                targetPdf[i] = ph[i];
            }
        }
    }
};
