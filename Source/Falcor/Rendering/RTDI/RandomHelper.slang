import Utils.Sampling.TinyUniformSampleGenerator;
typedef TinyUniformSampleGenerator RH_RandomSamplerState;
RH_RandomSamplerState RH_InitRandomState(uint2 pixel, uint stage, uint frameIndex) {
    return TinyUniformSampleGenerator(pixel, frameIndex + stage * 13);
}
float RH_GetNextRandom(inout RH_RandomSamplerState state) {
    return sampleNext1D(state);
}
float2 RH_GetNextRandom2D(inout RH_RandomSamplerState state) {
    return sampleNext2D(state);
}
