/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "ReSTIR_DI.h"
#include <algorithm>

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ReSTIR_DI>();
}

namespace
{
const char kInputVBuffer[] = "vbuffer";
const char kInputMVec[] = "mvec";
const char kOutputColor[] = "color";
const char kRISSampleCount[] = "risSampleCount";
const char kLocalRISSampleCount[] = "localRisSampleCount";
const char kInfiniteRISSampleCount[] = "infiniteRisSampleCount";
const char kEnvRISSampleCount[] = "envRisSampleCount";
const char kBrdfRISSampleCount[] = "brdfRisSampleCount";
const char kEnableTemporalReuse[] = "enableTemporalReuse";
const char kEnableSpatialReuse[] = "enableSpatialReuse";
const char kSpatialReuseCount[] = "spatialReuseCount";
const char kSpatialReuseSampleCount[] = "spatialReuseSampleCount";
const char kPresampledTileCount[] = "presampledTileCount";
const char kPresampledTileSize[] = "presampledTileSize";
const char kTemporalSamplingRadius[] = "temporalSamplingRadius";
const char kSpatialSamplingRadius[] = "spatialSamplingRadius";
const char kNormalThreshold[] = "normalThreshold";
const char kDepthThreshold[] = "depthThreshold";
const char kRayEpsilon[] = "rayEpsilon";
const char kUseEmissiveTextures[] = "useEmissiveTextures";
}

ReSTIR_DI::ReSTIR_DI(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    for (const auto& [key, value] : props)
    {
        if (key == kRISSampleCount)
            mOptions.localRisSampleCount = value;
        else if (key == kLocalRISSampleCount)
            mOptions.localRisSampleCount = value;
        else if (key == kInfiniteRISSampleCount)
            mOptions.infiniteRisSampleCount = value;
        else if (key == kEnvRISSampleCount)
            mOptions.envRisSampleCount = value;
        else if (key == kBrdfRISSampleCount)
            mOptions.brdfRisSampleCount = value;
        else if (key == kEnableTemporalReuse)
            mOptions.enableTemporalReuse = value;
        else if (key == kEnableSpatialReuse)
            mOptions.enableSpatialReuse = value;
        else if (key == kSpatialReuseCount)
            mOptions.spatialReuseCount = value;
        else if (key == kSpatialReuseSampleCount)
            mOptions.spatialReuseSampleCount = value;
        else if (key == kPresampledTileCount)
            mOptions.presampledTileCount = value;
        else if (key == kPresampledTileSize)
            mOptions.presampledTileSize = value;
        else if (key == kTemporalSamplingRadius)
            mOptions.temporalSamplingRadius = value;
        else if (key == kSpatialSamplingRadius)
            mOptions.spatialSamplingRadius = value;
        else if (key == kNormalThreshold)
            mOptions.normalThreshold = value;
        else if (key == kDepthThreshold)
            mOptions.depthThreshold = value;
        else if (key == kRayEpsilon)
            mOptions.rayEpsilon = value;
        else if (key == kUseEmissiveTextures)
            mOptions.useEmissiveTextures = value;
        else
            logWarning("Unknown property '{}' in ReSTIR_DI properties.", key);
    }

    mOptions.localRisSampleCount = std::min(mOptions.localRisSampleCount, 64u);
    mOptions.infiniteRisSampleCount = std::min(mOptions.infiniteRisSampleCount, 64u);
    mOptions.envRisSampleCount = std::min(mOptions.envRisSampleCount, 64u);
    mOptions.brdfRisSampleCount = std::min(mOptions.brdfRisSampleCount, 64u);
    mOptions.spatialReuseCount = std::max(mOptions.spatialReuseCount, 1u);
    mOptions.spatialReuseSampleCount = std::clamp(mOptions.spatialReuseSampleCount, 1u, 32u);
    mOptions.presampledTileCount = std::clamp(mOptions.presampledTileCount, 1u, 1024u);
    mOptions.presampledTileSize = std::clamp(mOptions.presampledTileSize, 1u, 8192u);
    mOptions.temporalSamplingRadius = std::clamp(mOptions.temporalSamplingRadius, 0.f, 64.f);
    mOptions.spatialSamplingRadius = std::clamp(mOptions.spatialSamplingRadius, 0.f, 64.f);
    mOptions.normalThreshold = std::clamp(mOptions.normalThreshold, 0.f, 1.f);
    mOptions.depthThreshold = std::clamp(mOptions.depthThreshold, 0.f, 1.f);
    mOptions.rayEpsilon = std::max(mOptions.rayEpsilon, 1.0e-6f);
}

Properties ReSTIR_DI::getProperties() const
{
    Properties props;
    props[kLocalRISSampleCount] = mOptions.localRisSampleCount;
    props[kInfiniteRISSampleCount] = mOptions.infiniteRisSampleCount;
    props[kEnvRISSampleCount] = mOptions.envRisSampleCount;
    props[kBrdfRISSampleCount] = mOptions.brdfRisSampleCount;
    props[kEnableTemporalReuse] = mOptions.enableTemporalReuse;
    props[kEnableSpatialReuse] = mOptions.enableSpatialReuse;
    props[kSpatialReuseCount] = mOptions.spatialReuseCount;
    props[kSpatialReuseSampleCount] = mOptions.spatialReuseSampleCount;
    props[kPresampledTileCount] = mOptions.presampledTileCount;
    props[kPresampledTileSize] = mOptions.presampledTileSize;
    props[kTemporalSamplingRadius] = mOptions.temporalSamplingRadius;
    props[kSpatialSamplingRadius] = mOptions.spatialSamplingRadius;
    props[kNormalThreshold] = mOptions.normalThreshold;
    props[kDepthThreshold] = mOptions.depthThreshold;
    props[kRayEpsilon] = mOptions.rayEpsilon;
    props[kUseEmissiveTextures] = mOptions.useEmissiveTextures;
    return props;
}

RenderPassReflection ReSTIR_DI::reflect(const CompileData& /*compileData*/)
{
    RenderPassReflection reflector;
    reflector.addInput(kInputVBuffer, "Visibility buffer in packed format").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addInput(kInputMVec, "Motion vector buffer (float2, current-to-previous in screen space)").bindFlags(ResourceBindFlags::ShaderResource);
    reflector.addOutput(kOutputColor, "Shaded output color")
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource)
        .format(ResourceFormat::RGBA32Float);
    return reflector;
}

void ReSTIR_DI::compile(RenderContext* /*pRenderContext*/, const CompileData& /*compileData*/)
{
}

void ReSTIR_DI::setScene(RenderContext* /*pRenderContext*/, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mpRTDI = nullptr;

    if (mpScene)
        mpRTDI = std::make_unique<RTDI>(mpScene, mOptions);
}

void ReSTIR_DI::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene || !mpRTDI)
        return;

    const auto pVBuffer = renderData.getTexture(kInputVBuffer);
    const auto pMotionVectors = renderData.getTexture(kInputMVec);
    const auto pColor = renderData.getTexture(kOutputColor);

    mpRTDI->execute(pRenderContext, pVBuffer, pMotionVectors, pColor);
}

void ReSTIR_DI::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;
    dirty |= widget.var("Local RIS M", mOptions.localRisSampleCount, 0u, 64u);
    dirty |= widget.var("Infinite RIS M", mOptions.infiniteRisSampleCount, 0u, 64u);
    dirty |= widget.var("Env RIS M", mOptions.envRisSampleCount, 0u, 64u);
    dirty |= widget.var("BRDF RIS M", mOptions.brdfRisSampleCount, 0u, 64u);
    dirty |= widget.checkbox("Enable Temporal Reuse", mOptions.enableTemporalReuse);
    dirty |= widget.checkbox("Enable Spatial Reuse", mOptions.enableSpatialReuse);
    dirty |= widget.var("Spatial Reuse times", mOptions.spatialReuseCount, 1u, 5u);
    dirty |= widget.var("Spatial Reuse Samples", mOptions.spatialReuseSampleCount, 1u, 32u);
    dirty |= widget.var("Presampled Tile Count", mOptions.presampledTileCount, 1u, 1024u);
    dirty |= widget.var("Presampled Tile Size", mOptions.presampledTileSize, 1u, 8192u);
    dirty |= widget.var("Temporal Radius", mOptions.temporalSamplingRadius, 0.f, 64.f, 0.1f);
    dirty |= widget.var("Spatial Radius", mOptions.spatialSamplingRadius, 0.f, 64.f, 0.1f);
    dirty |= widget.var("Normal Threshold", mOptions.normalThreshold, 0.f, 1.f, 0.001f);
    dirty |= widget.var("Depth Threshold", mOptions.depthThreshold, 0.f, 1.f, 0.001f);
    dirty |= widget.var("Ray Epsilon", mOptions.rayEpsilon, 1.0e-6f, 1.0e-2f, 1.0e-6f);
    dirty |= widget.checkbox("Use Emissive Textures", mOptions.useEmissiveTextures);

    if (dirty && mpRTDI)
        mpRTDI->setOptions(mOptions);
}
