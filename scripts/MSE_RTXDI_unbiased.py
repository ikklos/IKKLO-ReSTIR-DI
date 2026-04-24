from pathlib import WindowsPath, PosixPath
from falcor import *

def render_graph_MSE_RTXDI_unbiased():
    g = RenderGraph('MSE_RTXDI_unbiased')
    g.create_pass('ErrorMeasurePass', 'ErrorMeasurePass', {'ReferenceImagePath': 'reference_flipped.pfm', 'MeasurementsFilePath': '', 'IgnoreBackground': True, 'ComputeSquaredDifference': True, 'ComputeAverage': False, 'UseLoadedReference': True, 'ReportRunningError': True, 'RunningErrorSigma': 0.9950000047683716, 'SelectedOutputId': 'Source'})
    g.create_pass('VBufferRT', 'VBufferRT', {'outputSize': 'Default', 'samplePattern': 'Center', 'sampleCount': 16, 'useAlphaTest': True, 'adjustShadingNormals': True, 'forceCullMode': False, 'cull': 'Back', 'useTraceRayInline': False, 'useDOF': True})
    g.create_pass('RTXDIPass', 'RTXDIPass', {'options': {'mode': 'SpatiotemporalResampling', 'presampledTileCount': 128, 'presampledTileSize': 1024, 'storeCompactLightInfo': True, 'localLightCandidateCount': 24, 'infiniteLightCandidateCount': 8, 'envLightCandidateCount': 8, 'brdfCandidateCount': 1, 'brdfCutoff': 0.0, 'testCandidateVisibility': True, 'biasCorrection': 'Basic', 'depthThreshold': 0.10000000149011612, 'normalThreshold': 0.5, 'samplingRadius': 30.0, 'spatialSampleCount': 1, 'spatialIterations': 5, 'maxHistoryLength': 20, 'boilingFilterStrength': 0.0, 'rayEpsilon': 0.0010000000474974513, 'useEmissiveTextures': False, 'enableVisibilityShortcut': False, 'enablePermutationSampling': False}})
    g.add_edge('VBufferRT.mvec', 'RTXDIPass.mvec')
    g.add_edge('RTXDIPass.color', 'ErrorMeasurePass.Source')
    g.add_edge('VBufferRT.vbuffer', 'RTXDIPass.vbuffer')
    g.mark_output('ErrorMeasurePass.Output')
    return g

MSE_RTXDI_unbiased = render_graph_MSE_RTXDI_unbiased()
try: m.addGraph(MSE_RTXDI_unbiased)
except NameError: None
