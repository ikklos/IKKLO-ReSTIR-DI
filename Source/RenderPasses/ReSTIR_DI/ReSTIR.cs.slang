import Rendering.RTDI.RTDI;
import Rendering.RTDI.Reservoir;
[numthreads(256, 1, 1)]
void ReSTIR_Presample(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gRTDI.presampleLocalLights(dispatchThreadId.x);
}

[numthreads(256, 1, 1)]
void ReSTIR_PresampleEnv(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gRTDI.presampleEnvLights(dispatchThreadId.x);
}

[numthreads(16, 16, 1)]
void ReSTIR_Init(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gRTDI.init(dispatchThreadId.xy);
}

[numthreads(256, 1, 1)]
void ReSTIR_ClearReservoir(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint idx = dispatchThreadId.x;
    if (idx >= gRTDI.gFrameDim.x * gRTDI.gFrameDim.y) return;
    gRTDI.gReservoir[idx] = Reservoir();
}

[numthreads(16, 16, 1)]
void ReSTIR_Visibility(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gRTDI.visibility(dispatchThreadId.xy);
}

[numthreads(16, 16, 1)]
void ReSTIR_TemporalReuse(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gRTDI.temporalReuse(dispatchThreadId.xy, 16);
}

[numthreads(16, 16, 1)]
void ReSTIR_SpatialReuse(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gRTDI.spatialReuse(dispatchThreadId.xy, 9, 4.f);
}

[numthreads(16, 16, 1)]
void ReSTIR_ShadeColor(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gRTDI.shadeColor(dispatchThreadId.xy);
}
