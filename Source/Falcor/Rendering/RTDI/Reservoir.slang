import RandomHelper;
struct Reservoir
{
    static const uint kLightIndexMask = 0x7fffffffu;
    static const uint kLightValidBit = 0x80000000u;

    uint3 lightID = uint3(0u, 0u, 0u);
    float2 uv[3] = {};
    float3 wsum = 0.0;
    float3 targetPdf = 0.0;
    float M = 0.0;

    bool isValid(uint channel)
    {
        return channel < 3u && ((lightID[channel] & kLightValidBit) != 0u);
    }

    int getLightIndex(uint channel)
    {
        return isValid(channel) ? int(lightID[channel] & kLightIndexMask) : -1;
    }

    [mutating] void setLight(uint channel, int light)
    {
        if (channel >= 3u) return;
        lightID[channel] = (light >= 0) ? (kLightValidBit | (uint(light) & kLightIndexMask)) : 0u;
    }

    [mutating] void clearLight(uint channel)
    {
        if (channel >= 3u) return;
        lightID[channel] = 0u;
    }

    [mutating] void update(int light, float2 uv, float3 w, float3 ph, float m, inout RH_RandomSamplerState state)
    {
        wsum = wsum + w;
        M = M + m;
        [unroll]
        for (uint i = 0; i < 3; i++) {
            if (RH_GetNextRandom(state) < (w[i] / max(wsum[i], 1e-6f)))
            {
                setLight(i, light);
                this.uv[i] = uv;
                targetPdf[i] = ph[i];
            }
        }
    }
};
