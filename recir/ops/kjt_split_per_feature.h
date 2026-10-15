// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/string_view.h>

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

namespace facebook::recir {

std::vector<at::Tensor> kjtSplitPerFeature(
    const at::Tensor& values,
    const at::Tensor& offsets,
    int64_t stride,
    int64_t numKeys);

std::vector<at::Tensor> kjtLengthsPerFeature(
    const at::Tensor& offsets,
    int64_t numKeys);

at::Tensor concat2DJagged(
    int64_t maxSeqLen,
    const at::Tensor& valuesA,
    const at::Tensor& valuesB,
    const std::optional<at::Tensor>& offsetsA,
    const std::optional<at::Tensor>& offsetsB,
    bool isReplace,
    int64_t numPrefixFromRight);

std::tuple<at::Tensor, at::Tensor, at::Tensor> weightedLayerNorm(
    const at::Tensor& x,
    const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias,
    double epsilon,
    const std::optional<at::Tensor>& mean,
    const std::optional<at::Tensor>& reciprocalStd,
    bool isSwish);

at::Tensor tritonAddMm(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& bias);

std::tuple<at::Tensor, int64_t, int64_t, int64_t> jaggedDenseBmmAdd(
    int64_t maxSeqLen,
    const at::Tensor& sequenceOffsets,
    const at::Tensor& jagged,
    const at::Tensor& dense,
    const at::Tensor& bias,
    bool elementwise);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
addTimestampPositionEmbeddings(
    const at::Tensor& sequenceEmbeddings,
    const at::Tensor& sequenceOffsets,
    const at::Tensor& positionEmbeddings,
    const at::Tensor& timestampEmbeddings,
    const at::Tensor& timestamps,
    const at::Tensor& sequenceLengths,
    const std::optional<at::Tensor>& numTargets,
    int64_t maxSeqLen,
    int64_t maxContextualSeqLen,
    bool interleaveTargets,
    c10::string_view timeBucketFunction);

std::tuple<at::Tensor, at::Tensor> split2DJagged(
    const at::Tensor& values,
    int64_t maxSeqLen,
    const at::Tensor& offsetsA,
    const at::Tensor& offsetsB,
    int64_t sequenceLengthA,
    int64_t sequenceLengthB,
    bool isDenseA,
    bool isDenseB,
    int64_t denseSize,
    int64_t numPrefixToRight,
    int64_t batchSize);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
weightedLayerNormQuantized(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& bias,
    double epsilon,
    const at::Tensor& mean,
    const at::Tensor& reciprocalStd,
    int64_t blockDimension);

at::Tensor fp8RowwiseAddMm(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& aScale,
    const at::Tensor& bScale,
    const at::Tensor& bias,
    bool hasBias,
    bool biasIs2D,
    at::ScalarType outputType,
    bool allowTf32,
    bool useFastAccumulation);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
layerNormMulQuantized(
    const at::Tensor& x,
    const at::Tensor& u,
    const at::Tensor& weight,
    const at::Tensor& bias,
    double epsilon,
    double dropoutRatio,
    bool training,
    bool siluU,
    bool concatU,
    bool concatX,
    c10::string_view multiplyActivation,
    int64_t seed,
    bool saveOutput,
    int64_t blockDimension);

std::tuple<at::Tensor, at::Tensor> raggedAttention(
    int64_t maxSeqLen,
    double alpha,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& sequenceOffsets,
    c10::string_view invalidAttentionMaskType,
    const std::optional<at::Tensor>& numTargets,
    const std::optional<at::Tensor>& attentionScale,
    int64_t maxAttentionLength,
    int64_t contextualSequenceLength,
    int64_t fullAttentionSize,
    int64_t numSoftmaxHeads);

} // namespace facebook::recir
