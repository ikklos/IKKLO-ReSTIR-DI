import RandomHelper;

struct FinReservoir
{
    static const uint kLightIndexMask = 0x7fffffffu;
    static const uint kLightValidBit = 0x80000000u;

    uint lightID = 0u;
    float2 uv = 0.f;
    float wsum = 0.f;
    float targetPdf = 0.f;
    float M = 0.f;

    bool isValid()
    {
        return (lightID & kLightValidBit) != 0u;
    }

    int getLightIndex()
    {
        return isValid() ? int(lightID & kLightIndexMask) : -1;
    }

    [mutating] void setLight(int light)
    {
        lightID = (light >= 0) ? (kLightValidBit | (uint(light) & kLightIndexMask)) : 0u;
    }

    [mutating] void clearLight()
    {
        lightID = 0u;
    }

    [mutating] void update(int light, float2 uv, float w, float ph, float m, inout RH_RandomSamplerState state)
    {
        wsum += w;
        M += m;
        if (RH_GetNextRandom(state) < (w / max(wsum, 1e-6f)))
        {
            setLight(light);
            this.uv = uv;
            targetPdf = ph;
        }
    }
};
