from pathlib import WindowsPath, PosixPath
from falcor import *

def render_graph_ReSTIRDI():
    g = RenderGraph('ReSTIRDI')
    g.create_pass('VBufferRT', 'VBufferRT', {'outputSize': 'Default', 'samplePattern': 'Center', 'sampleCount': 16, 'useAlphaTest': True, 'adjustShadingNormals': True, 'forceCullMode': False, 'cull': 'Back', 'useTraceRayInline': False, 'useDOF': True})
    g.create_pass(
        'ReSTIR_DI',
        'ReSTIR_DI',
        {
            'localRisSampleCount': 24,
            'infiniteRisSampleCount': 8,
            'envRisSampleCount': 8,
            'brdfRisSampleCount': 1,
            'spatialReuseCount': 1,
        },
    )
    g.create_pass('ToneMapper', 'ToneMapper', {'outputSize': 'Default', 'useSceneMetadata': True, 'exposureCompensation': 0.0, 'autoExposure': False, 'filmSpeed': 100.0, 'whiteBalance': False, 'whitePoint': 6500.0, 'operator': 'Aces', 'clamp': True, 'whiteMaxLuminance': 1.0, 'whiteScale': 11.199999809265137, 'fNumber': 1.0, 'shutter': 1.0, 'exposureMode': 'AperturePriority'})
    g.create_pass('AccumulatePass', 'AccumulatePass', {'enabled': True, 'outputSize': 'Default', 'autoReset': True, 'precisionMode': 'Single', 'maxFrameCount': 0, 'overflowMode': 'Stop'})
    g.add_edge('VBufferRT.vbuffer', 'ReSTIR_DI.vbuffer')
    g.add_edge('VBufferRT.mvec', 'ReSTIR_DI.mvec')
    g.add_edge('ReSTIR_DI.color', 'AccumulatePass.input')
    g.add_edge('AccumulatePass.output', 'ToneMapper.src')
    g.mark_output('ToneMapper.dst')
    return g

ReSTIRDI = render_graph_ReSTIRDI()
try: m.addGraph(ReSTIRDI)
except NameError: None
