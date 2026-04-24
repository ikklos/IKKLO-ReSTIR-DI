from pathlib import WindowsPath, PosixPath
from falcor import *

def render_graph_MSE_VecReSTIR_unbiased():
    g = RenderGraph('MSE_VecReSTIR_unbiased')
    g.create_pass('ErrorMeasurePass', 'ErrorMeasurePass', {'ReferenceImagePath': 'reference_flipped.pfm', 'MeasurementsFilePath': '', 'IgnoreBackground': True, 'ComputeSquaredDifference': True, 'ComputeAverage': False, 'UseLoadedReference': True, 'ReportRunningError': True, 'RunningErrorSigma': 0.9950000047683716, 'SelectedOutputId': 'Source'})
    g.create_pass('VBufferRT', 'VBufferRT', {'outputSize': 'Default', 'samplePattern': 'Center', 'sampleCount': 16, 'useAlphaTest': True, 'adjustShadingNormals': True, 'forceCullMode': False, 'cull': 'Back', 'useTraceRayInline': False, 'useDOF': True})
    g.create_pass('ReSTIR_DI', 'ReSTIR_DI', {'localRisSampleCount': 24, 'infiniteRisSampleCount': 8, 'envRisSampleCount': 8, 'brdfRisSampleCount': 1, 'enableTemporalReuse': True, 'enableSpatialReuse': True, 'spatialReuseCount': 1, 'spatialReuseSampleCount': 1, 'presampledTileCount': 128, 'presampledTileSize': 1024, 'temporalSamplingRadius': 4.0, 'spatialSamplingRadius': 30.0, 'normalThreshold': 0.5, 'depthThreshold': 0.10000000149011612, 'rayEpsilon': 0.0010000000474974513, 'useEmissiveTextures': 0})
    g.add_edge('VBufferRT.vbuffer', 'ReSTIR_DI.vbuffer')
    g.add_edge('ReSTIR_DI.color', 'ErrorMeasurePass.Source')
    g.add_edge('VBufferRT.mvec', 'ReSTIR_DI.mvec')
    g.mark_output('ErrorMeasurePass.Output')
    return g

MSE_VecReSTIR_unbiased = render_graph_MSE_VecReSTIR_unbiased()
try: m.addGraph(MSE_VecReSTIR_unbiased)
except NameError: None
