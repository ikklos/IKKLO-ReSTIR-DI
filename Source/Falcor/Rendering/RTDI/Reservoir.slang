import RandomHelper;
struct Reservoir
{
    int lightID = -1;
    float2 sampleUV = float2(0, 0);

    float W = 0.0;
    float wsum = 0.0;
    float targetPdf = 0.0;
    float M = 0;

    [mutating] void update(int light, float2 uv, float w, float ph, float m, inout RH_RandomSamplerState state)
    {
        wsum = wsum + w;
        M = M + uint(m);
        if (RH_GetNextRandom(state) < (w / max(wsum, 1e-6f)))
        {
            lightID = light;
            sampleUV = uv;
            targetPdf = ph;
        }
    }
};
