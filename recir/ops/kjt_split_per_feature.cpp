// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "recir/ops/kjt_split_per_feature.h"

#include <ATen/ATen.h>
#include <torch/library.h>

namespace facebook::recir {
namespace {

struct JaggedRange {
  int64_t start;
  int64_t end;
};

JaggedRange getJaggedRange(
    const at::Tensor& cpuOffsets,
    int64_t batch,
    int64_t valuesSize) {
  const auto start = cpuOffsets.select(0, batch).item<int64_t>();
  const auto end = cpuOffsets.select(0, batch + 1).item<int64_t>();
  TORCH_CHECK(
      0 <= start && start <= end && end <= valuesSize,
      "invalid jagged range [",
      start,
      ", ",
      end,
      ") for values with size ",
      valuesSize);
  return {start, end};
}

at::Tensor getBatchValues(
    const at::Tensor& values,
    const std::optional<at::Tensor>& cpuOffsets,
    int64_t batch) {
  if (!cpuOffsets.has_value()) {
    return values.select(0, batch);
  }
  const auto range = getJaggedRange(*cpuOffsets, batch, values.size(0));
  return values.slice(0, range.start, range.end);
}

} // namespace

std::vector<at::Tensor> kjtSplitPerFeature(
    const at::Tensor& values,
    const at::Tensor& offsets,
    int64_t stride,
    int64_t numKeys) {
  TORCH_CHECK(stride > 0, "stride must be positive, got ", stride);
  TORCH_CHECK(numKeys >= 0, "num_keys must be non-negative, got ", numKeys);
  TORCH_CHECK(offsets.dim() == 1, "offsets must be one-dimensional");
  TORCH_CHECK(
      numKeys == 0 || numKeys * stride < offsets.numel(),
      "offsets does not contain a terminal offset for every key");

  const auto cpuOffsets = offsets.cpu().contiguous();
  std::vector<at::Tensor> result;
  result.reserve(numKeys);
  for (int64_t key = 0; key < numKeys; ++key) {
    const auto start = cpuOffsets.select(0, key * stride).item<int64_t>();
    const auto end = cpuOffsets.select(0, (key + 1) * stride).item<int64_t>();
    TORCH_CHECK(
        0 <= start && start <= end && end <= values.size(0),
        "invalid value range [",
        start,
        ", ",
        end,
        ") for values with size ",
        values.size(0));
    result.push_back(values.slice(0, start, end).clone());
  }
  return result;
}

std::vector<at::Tensor> kjtLengthsPerFeature(
    const at::Tensor& offsets,
    int64_t numKeys) {
  TORCH_CHECK(numKeys > 0, "num_keys must be positive, got ", numKeys);
  TORCH_CHECK(offsets.dim() == 1, "offsets must be one-dimensional");
  TORCH_CHECK(offsets.numel() > 0, "offsets must contain a terminal offset");
  TORCH_CHECK(
      (offsets.numel() - 1) % numKeys == 0,
      "offsets must contain the same runtime stride for every key");

  const auto runtimeStride = (offsets.numel() - 1) / numKeys;
  const auto flatLengths =
      offsets.slice(0, 1) - offsets.slice(0, 0, offsets.numel() - 1);
  std::vector<at::Tensor> result;
  result.reserve(numKeys);
  for (int64_t key = 0; key < numKeys; ++key) {
    result.push_back(
        flatLengths.slice(0, key * runtimeStride, (key + 1) * runtimeStride)
            .clone());
  }
  return result;
}

at::Tensor concat2DJagged(
    int64_t maxSeqLen,
    const at::Tensor& valuesA,
    const at::Tensor& valuesB,
    const std::optional<at::Tensor>& offsetsA,
    const std::optional<at::Tensor>& offsetsB,
    bool isReplace,
    int64_t numPrefixFromRight) {
  static_cast<void>(maxSeqLen);
  TORCH_CHECK(
      offsetsA.has_value() || offsetsB.has_value(),
      "at least one input must be jagged");
  TORCH_CHECK(
      numPrefixFromRight >= 0,
      "n_prefix_from_right must be non-negative, got ",
      numPrefixFromRight);
  TORCH_CHECK(
      !isReplace || numPrefixFromRight == 0,
      "n_prefix_from_right is not supported when is_replace is true");
  TORCH_CHECK(
      numPrefixFromRight == 0 || (offsetsA.has_value() && offsetsB.has_value()),
      "n_prefix_from_right is only supported for two jagged inputs");
  TORCH_CHECK(
      valuesA.device() == valuesB.device(),
      "values_a and values_b must be on the same device");
  TORCH_CHECK(
      valuesA.scalar_type() == valuesB.scalar_type(),
      "values_a and values_b must have the same dtype");
  TORCH_CHECK(
      valuesA.size(-1) == valuesB.size(-1),
      "values_a and values_b must have the same embedding dimension");

  const auto cpuOffsetsA = offsetsA.has_value()
      ? std::make_optional(offsetsA->cpu().contiguous())
      : std::nullopt;
  const auto cpuOffsetsB = offsetsB.has_value()
      ? std::make_optional(offsetsB->cpu().contiguous())
      : std::nullopt;

  int64_t batchSize;
  if (!offsetsA.has_value()) {
    TORCH_CHECK(valuesA.dim() == 3, "dense values_a must be three-dimensional");
    batchSize = valuesA.size(0);
    TORCH_CHECK(
        offsetsB->numel() == batchSize + 1,
        "offsets_b size must match the dense batch size");
  } else if (!offsetsB.has_value()) {
    TORCH_CHECK(valuesB.dim() == 3, "dense values_b must be three-dimensional");
    batchSize = valuesB.size(0);
    TORCH_CHECK(
        offsetsA->numel() == batchSize + 1,
        "offsets_a size must match the dense batch size");
  } else {
    TORCH_CHECK(
        offsetsA->dim() == 1 && offsetsB->dim() == 1,
        "jagged offsets must be one-dimensional");
    TORCH_CHECK(
        offsetsA->numel() == offsetsB->numel(),
        "offsets_a and offsets_b must have the same size");
    TORCH_CHECK(offsetsA->numel() > 0, "jagged offsets cannot be empty");
    batchSize = offsetsA->numel() - 1;
  }

  std::vector<at::Tensor> batches;
  batches.reserve(batchSize);
  for (int64_t batch = 0; batch < batchSize; ++batch) {
    const auto batchA = getBatchValues(valuesA, cpuOffsetsA, batch);
    const auto batchB = getBatchValues(valuesB, cpuOffsetsB, batch);
    if (isReplace) {
      TORCH_CHECK(
          batchB.size(0) <= batchA.size(0),
          "replacement values_b cannot be longer than values_a");
      batches.push_back(
          at::cat(
              {batchA.slice(0, 0, batchA.size(0) - batchB.size(0)), batchB},
              0));
      continue;
    }

    TORCH_CHECK(
        numPrefixFromRight <= batchB.size(0),
        "n_prefix_from_right cannot exceed a values_b sequence length");
    batches.push_back(
        at::cat(
            {batchB.slice(0, 0, numPrefixFromRight),
             batchA,
             batchB.slice(0, numPrefixFromRight)},
            0));
  }

  const auto flatResult = batches.empty()
      ? at::empty({0, valuesA.size(-1)}, valuesA.options())
      : at::cat(batches, 0);
  return isReplace && valuesA.dim() == 3 ? flatResult.view_as(valuesA)
                                         : flatResult;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> weightedLayerNorm(
    const at::Tensor& x,
    const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias,
    double epsilon,
    const std::optional<at::Tensor>& mean,
    const std::optional<at::Tensor>& reciprocalStd,
    bool isSwish) {
  TORCH_CHECK(x.dim() == 2, "x must be two-dimensional");
  TORCH_CHECK(
      !isSwish || weight.has_value(),
      "is_swish requires learnable layer normalization");
  if (weight.has_value()) {
    TORCH_CHECK(bias.has_value(), "bias is required when weight is provided");
    TORCH_CHECK(
        weight->dim() == 1 && weight->numel() == x.size(1),
        "weight must match the feature dimension");
    TORCH_CHECK(
        bias->dim() == 1 && bias->numel() == x.size(1),
        "bias must match the feature dimension");
  }

  const auto xFloat = x.to(at::kFloat);
  at::Tensor outputMean;
  at::Tensor outputReciprocalStd;
  if (!mean.has_value() || !reciprocalStd.has_value()) {
    outputMean = xFloat.mean(1, false);
    const auto centered = xFloat - outputMean.unsqueeze(1);
    outputReciprocalStd =
        at::rsqrt((centered * centered).mean(1, false) + epsilon);
  } else {
    TORCH_CHECK(
        mean->numel() == x.size(0) && reciprocalStd->numel() == x.size(0),
        "mean and rstd must match the row count");
    outputMean = mean->to(at::kFloat).clone();
    outputReciprocalStd = reciprocalStd->to(at::kFloat).clone();
  }

  auto normalized =
      (xFloat - outputMean.unsqueeze(1)) * outputReciprocalStd.unsqueeze(1);
  if (weight.has_value()) {
    normalized = normalized * weight->to(at::kFloat) + bias->to(at::kFloat);
  }
  if (isSwish) {
    normalized = at::sigmoid(normalized) * xFloat;
  }
  return {normalized.to(x.scalar_type()), outputMean, outputReciprocalStd};
}

at::Tensor tritonAddMm(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& bias) {
  TORCH_CHECK(
      x.dim() == 2 && weight.dim() == 2,
      "x and weight must be two-dimensional");
  TORCH_CHECK(
      x.size(1) == weight.size(0), "x and weight dimensions are incompatible");
  TORCH_CHECK(
      (bias.dim() == 1 && bias.size(0) == weight.size(1)) ||
          (bias.dim() == 2 && bias.size(1) == weight.size(1)),
      "bias dimensions are incompatible with the output");
  return at::addmm(bias, x, weight);
}

std::tuple<at::Tensor, int64_t, int64_t, int64_t> jaggedDenseBmmAdd(
    int64_t maxSeqLen,
    const at::Tensor& sequenceOffsets,
    const at::Tensor& jagged,
    const at::Tensor& dense,
    const at::Tensor& bias,
    bool elementwise) {
  static_cast<void>(maxSeqLen);
  TORCH_CHECK(jagged.dim() == 2, "jagged must be two-dimensional");
  TORCH_CHECK(dense.dim() == 3, "dense must be three-dimensional");
  TORCH_CHECK(
      sequenceOffsets.dim() == 1, "sequence offsets must be one-dimensional");
  const auto batchSize = dense.size(0);
  const auto inputDimension = jagged.size(1);
  const auto outputDimension = dense.size(2);
  TORCH_CHECK(
      dense.size(1) == inputDimension,
      "jagged and dense input dimensions are incompatible");
  TORCH_CHECK(
      sequenceOffsets.numel() == batchSize + 1,
      "sequence offsets size must match the dense batch size");
  TORCH_CHECK(
      elementwise
          ? bias.sizes() == at::IntArrayRef({jagged.size(0), outputDimension})
          : bias.sizes() == at::IntArrayRef({batchSize, outputDimension}),
      "bias dimensions are incompatible with the output");

  const auto cpuOffsets = sequenceOffsets.cpu().contiguous();
  std::vector<at::Tensor> batches;
  batches.reserve(batchSize);
  for (int64_t batch = 0; batch < batchSize; ++batch) {
    const auto range = getJaggedRange(cpuOffsets, batch, jagged.size(0));
    const auto jaggedBatch = jagged.slice(0, range.start, range.end);
    const auto biasBatch = elementwise ? bias.slice(0, range.start, range.end)
                                       : bias.select(0, batch);
    batches.push_back(
        at::addmm(biasBatch, jaggedBatch, dense.select(0, batch)));
  }

  const auto output = batches.empty()
      ? at::empty({0, outputDimension}, jagged.options())
      : at::cat(batches, 0);
  return {output, batchSize, inputDimension, outputDimension};
}

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
    c10::string_view timeBucketFunction) {
  static_cast<void>(maxSeqLen);
  TORCH_CHECK(
      sequenceEmbeddings.dim() == 2,
      "sequence embeddings must be two-dimensional");
  TORCH_CHECK(
      positionEmbeddings.dim() == 2 && timestampEmbeddings.dim() == 2,
      "position and timestamp embeddings must be two-dimensional");
  TORCH_CHECK(timestamps.dim() == 2, "timestamps must be two-dimensional");
  TORCH_CHECK(
      sequenceLengths.dim() == 1, "sequence lengths must be one-dimensional");
  TORCH_CHECK(
      sequenceEmbeddings.size(1) == positionEmbeddings.size(1) &&
          sequenceEmbeddings.size(1) == timestampEmbeddings.size(1),
      "all embeddings must have the same dimension");
  TORCH_CHECK(
      timestampEmbeddings.size(0) >= 2049,
      "timestamp embeddings must contain 2049 buckets");

  const auto batchSize = sequenceLengths.numel();
  TORCH_CHECK(
      sequenceOffsets.numel() == batchSize + 1,
      "sequence offsets size must match the batch size");
  TORCH_CHECK(
      timestamps.size(0) == batchSize,
      "timestamps size must match the batch size");
  TORCH_CHECK(
      !numTargets.has_value() || numTargets->numel() == batchSize,
      "num_targets size must match the batch size");

  const auto cpuOffsets = sequenceOffsets.cpu().contiguous();
  const auto cpuLengths = sequenceLengths.cpu().contiguous();
  const auto cpuNumTargets = numTargets.has_value()
      ? std::make_optional(numTargets->cpu().contiguous())
      : std::nullopt;
  std::vector<at::Tensor> outputBatches;
  std::vector<at::Tensor> positionIndexBatches;
  std::vector<at::Tensor> timestampIndexBatches;
  outputBatches.reserve(batchSize);
  positionIndexBatches.reserve(batchSize);
  timestampIndexBatches.reserve(batchSize);

  const auto indexOptions = sequenceEmbeddings.options().dtype(at::kLong);
  for (int64_t batch = 0; batch < batchSize; ++batch) {
    const auto range =
        getJaggedRange(cpuOffsets, batch, sequenceEmbeddings.size(0));
    const auto jaggedLength = range.end - range.start;
    const auto sequenceLength = cpuLengths.select(0, batch).item<int64_t>();
    TORCH_CHECK(
        sequenceLength > 0 && sequenceLength <= timestamps.size(1),
        "sequence length is outside the timestamp range");

    auto highIndex = sequenceLength;
    if (cpuNumTargets.has_value()) {
      const auto targets = cpuNumTargets->select(0, batch).item<int64_t>();
      highIndex -= targets * (interleaveTargets ? 2 : 1);
    }
    TORCH_CHECK(highIndex >= 0, "target count exceeds the sequence length");

    const auto sequenceIndices = at::arange(jaggedLength, indexOptions);
    auto positionIndices = highIndex -
        at::clamp_max(sequenceIndices, highIndex) + maxContextualSeqLen;
    positionIndices =
        at::clamp_max(positionIndices, positionEmbeddings.size(0) - 1);
    positionIndices = at::where(
        sequenceIndices < maxContextualSeqLen,
        sequenceIndices,
        positionIndices);

    const auto batchTimestamps = timestamps.select(0, batch);
    auto timestampDeltas = (batchTimestamps.select(0, sequenceLength - 1) -
                            batchTimestamps.slice(0, 0, jaggedLength))
                               .to(at::kFloat);
    timestampDeltas = at::clamp_min(timestampDeltas, 1e-6) / 60.0;
    timestampDeltas = timeBucketFunction == "log" ? at::log(timestampDeltas)
                                                  : at::sqrt(timestampDeltas);
    const auto timestampIndices =
        at::clamp(timestampDeltas.to(at::kLong), 0, 2048);

    const auto sequenceBatch =
        sequenceEmbeddings.slice(0, range.start, range.end);
    const auto addedEmbeddings =
        (positionEmbeddings.index_select(0, positionIndices) +
         timestampEmbeddings.index_select(0, timestampIndices))
            .to(sequenceEmbeddings.scalar_type());
    outputBatches.push_back(sequenceBatch + addedEmbeddings);
    positionIndexBatches.push_back(positionIndices.to(at::kInt));
    timestampIndexBatches.push_back(timestampIndices.to(at::kInt));
  }

  const auto emptyIndex =
      at::empty({0}, sequenceEmbeddings.options().dtype(at::kInt));
  const auto output = outputBatches.empty() ? at::empty_like(sequenceEmbeddings)
                                            : at::cat(outputBatches, 0);
  const auto positionIndices = positionIndexBatches.empty()
      ? emptyIndex
      : at::cat(positionIndexBatches, 0);
  const auto timestampIndices = timestampIndexBatches.empty()
      ? emptyIndex
      : at::cat(timestampIndexBatches, 0);
  const auto [sortedPositionIndices, positionOrder] = at::sort(positionIndices);
  const auto [sortedTimestampIndices, timestampOrder] =
      at::sort(timestampIndices);
  return {
      output,
      sortedPositionIndices,
      positionOrder.to(at::kInt),
      sortedTimestampIndices,
      timestampOrder.to(at::kInt)};
}

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
    int64_t batchSize) {
  static_cast<void>(maxSeqLen);
  const auto cpuOffsetsA = offsetsA.cpu().contiguous();
  const auto cpuOffsetsB = offsetsB.cpu().contiguous();
  std::vector<at::Tensor> batchesA;
  std::vector<at::Tensor> batchesB;
  batchesA.reserve(batchSize);
  batchesB.reserve(batchSize);
  int64_t inputOffset = 0;
  for (int64_t batch = 0; batch < batchSize; ++batch) {
    const auto lengthA = isDenseA
        ? denseSize
        : getJaggedRange(cpuOffsetsA, batch, sequenceLengthA).end -
            getJaggedRange(cpuOffsetsA, batch, sequenceLengthA).start;
    const auto lengthB = isDenseB
        ? denseSize
        : getJaggedRange(cpuOffsetsB, batch, sequenceLengthB).end -
            getJaggedRange(cpuOffsetsB, batch, sequenceLengthB).start;
    TORCH_CHECK(
        numPrefixToRight <= lengthB,
        "n_prefix_to_right cannot exceed a values_b sequence length");
    const auto batchValues =
        values.slice(0, inputOffset, inputOffset + lengthA + lengthB);
    batchesA.push_back(
        batchValues.slice(0, numPrefixToRight, numPrefixToRight + lengthA));
    batchesB.push_back(
        at::cat(
            {batchValues.slice(0, 0, numPrefixToRight),
             batchValues.slice(0, numPrefixToRight + lengthA)},
            0));
    inputOffset += lengthA + lengthB;
  }
  TORCH_CHECK(
      inputOffset == values.size(0), "split lengths do not match values");

  auto valuesA = batchesA.empty()
      ? at::empty({0, values.size(1)}, values.options())
      : at::cat(batchesA, 0);
  auto valuesB = batchesB.empty()
      ? at::empty({0, values.size(1)}, values.options())
      : at::cat(batchesB, 0);
  if (isDenseA) {
    valuesA = valuesA.view({batchSize, denseSize, values.size(1)});
  }
  if (isDenseB) {
    valuesB = valuesB.view({batchSize, denseSize, values.size(1)});
  }
  return {valuesA, valuesB};
}

namespace {

std::pair<at::Tensor, at::Tensor> quantizeFp8Rowwise(const at::Tensor& values) {
  constexpr double kMaxFp8 = 448.0;
  const auto rowMaximum =
      std::get<0>(at::max(at::abs(values.to(at::kFloat)), 1, false));
  const auto scale = at::clamp_min(rowMaximum, 1e-12) / kMaxFp8;
  const auto quantized =
      at::clamp(values.to(at::kFloat) / scale.unsqueeze(1), -kMaxFp8, kMaxFp8)
          .to(at::kFloat8_e4m3fn);
  return {scale, quantized};
}

} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
weightedLayerNormQuantized(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& bias,
    double epsilon,
    const at::Tensor& mean,
    const at::Tensor& reciprocalStd,
    int64_t blockDimension) {
  static_cast<void>(blockDimension);
  const auto statistics = weightedLayerNorm(
      x,
      weight,
      bias,
      epsilon,
      mean.numel() == 0 ? std::nullopt : std::make_optional(mean),
      reciprocalStd.numel() == 0 ? std::nullopt
                                 : std::make_optional(reciprocalStd),
      false);
  const auto& output = std::get<0>(statistics);
  const auto [scale, quantized] = quantizeFp8Rowwise(output);
  return {
      output,
      std::get<1>(statistics),
      std::get<2>(statistics),
      scale,
      quantized};
}

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
    bool useFastAccumulation) {
  static_cast<void>(biasIs2D);
  static_cast<void>(allowTf32);
  static_cast<void>(useFastAccumulation);
  const auto dequantizedA = a.to(at::kFloat) * aScale.unsqueeze(1);
  const auto dequantizedB = b.to(at::kFloat) * bScale.unsqueeze(1);
  auto output = at::mm(dequantizedA, dequantizedB.transpose(0, 1));
  if (hasBias) {
    output = output + bias.to(at::kFloat);
  }
  return output.to(outputType);
}

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
    int64_t blockDimension) {
  static_cast<void>(dropoutRatio);
  static_cast<void>(training);
  static_cast<void>(seed);
  static_cast<void>(blockDimension);
  const auto [normalized, mean, reciprocalStd] = weightedLayerNorm(
      x, weight, bias, epsilon, std::nullopt, std::nullopt, false);
  auto multiplier = u;
  if (multiplyActivation == "silu") {
    multiplier = u * at::sigmoid(u);
  } else if (multiplyActivation == "sigmoid") {
    multiplier = at::sigmoid(u);
  }
  auto multiplied = normalized * multiplier;
  auto outputU = siluU ? u * at::sigmoid(u) : u;
  std::vector<at::Tensor> outputParts;
  if (concatU) {
    outputParts.push_back(outputU);
  }
  if (concatX) {
    outputParts.push_back(x);
  }
  outputParts.push_back(multiplied);
  const auto combined = at::cat(outputParts, 1);
  const auto [scale, quantized] = quantizeFp8Rowwise(combined);
  const auto output = saveOutput ? combined : at::empty({0}, x.options());
  return {output, mean, reciprocalStd, scale, quantized};
}

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
    int64_t numSoftmaxHeads) {
  TORCH_CHECK(numSoftmaxHeads == 0, "softmax attention heads are unsupported");
  TORCH_CHECK(
      query.dim() == 3 && key.dim() == 3 && value.dim() == 3,
      "query, key, and value must be three-dimensional");
  TORCH_CHECK(
      query.size(1) == key.size(1) && query.size(1) == value.size(1),
      "grouped-query attention is unsupported");
  const auto cpuOffsets = sequenceOffsets.cpu().contiguous();
  const auto cpuTargets = numTargets.has_value()
      ? std::make_optional(numTargets->cpu().contiguous())
      : std::nullopt;
  const auto batchSize = sequenceOffsets.numel() - 1;
  std::vector<at::Tensor> outputBatches;
  outputBatches.reserve(batchSize);
  const auto boolOptions = query.options().dtype(at::kBool);
  for (int64_t batch = 0; batch < batchSize; ++batch) {
    const auto range = getJaggedRange(cpuOffsets, batch, query.size(0));
    const auto length = range.end - range.start;
    const auto q = query.slice(0, range.start, range.end).transpose(0, 1);
    const auto k = key.slice(0, range.start, range.end).transpose(0, 1);
    const auto v = value.slice(0, range.start, range.end).transpose(0, 1);
    auto scores = at::matmul(q, k.transpose(1, 2)) * alpha;

    auto ids = at::arange(length, query.options().dtype(at::kLong));
    auto maskIds = ids;
    int64_t targetCount = 0;
    if (cpuTargets.has_value()) {
      targetCount = cpuTargets->select(0, batch).item<int64_t>();
      maskIds = at::clamp_max(ids, length - targetCount);
    }
    auto rows = maskIds.unsqueeze(1);
    auto columns = maskIds.unsqueeze(0);
    auto distance = rows - columns;
    auto mask = at::eye(length, boolOptions);
    const bool causal = invalidAttentionMaskType == "lower_triangular";
    if (!causal) {
      distance = at::abs(distance);
    }
    if (maxAttentionLength > 0) {
      mask = at::logical_or(
          mask,
          causal ? at::logical_and(distance > 0, distance <= maxAttentionLength)
                 : at::logical_and(
                       distance >= -maxAttentionLength,
                       distance <= maxAttentionLength));
      if (fullAttentionSize > 0) {
        const auto boundary = length - targetCount - fullAttentionSize;
        const auto globalRows =
            at::where(rows < boundary, at::zeros_like(rows), rows);
        const auto globalColumns =
            at::where(columns < boundary, at::zeros_like(columns), columns);
        mask = at::logical_or(mask, globalRows - globalColumns > 0);
      }
    } else if (causal) {
      mask = at::logical_or(mask, distance > 0);
    }
    if (contextualSequenceLength > 0) {
      mask = at::logical_or(mask, columns == 0);
    }

    auto weights = scores * at::sigmoid(scores);
    if (attentionScale.has_value()) {
      const auto scale = attentionScale->dim() == 0
          ? *attentionScale
          : attentionScale->slice(0, range.start, range.end)
                .view({1, length, 1});
      weights = weights * scale.to(weights.scalar_type());
    } else {
      weights = weights / maxSeqLen;
    }
    weights = weights * mask.unsqueeze(0).to(weights.scalar_type());
    outputBatches.push_back(at::matmul(weights, v).transpose(0, 1));
  }
  const auto output = outputBatches.empty()
      ? at::empty({0, query.size(1), value.size(2)}, value.options())
      : at::cat(outputBatches, 0);
  return {output, at::zeros({1}, query.options().dtype(at::kFloat))};
}

} // namespace facebook::recir

extern "C" __attribute__((used, retain, visibility("default"))) void
recir_kjt_split_per_feature_registration_anchor() {}

TORCH_LIBRARY_FRAGMENT(recir_torchrec, m) {
  m.def(
      torch::schema(
          "kjt_split_per_feature(Tensor values, Tensor offsets, int stride, int num_keys) -> Tensor[]",
          c10::AliasAnalysisKind::PURE_FUNCTION));
  m.def(
      torch::schema(
          "kjt_lengths_per_feature(Tensor offsets, int num_keys) -> Tensor[]",
          c10::AliasAnalysisKind::PURE_FUNCTION));
}

TORCH_LIBRARY_IMPL(recir_torchrec, CompositeImplicitAutograd, m) {
  m.impl(
      "kjt_split_per_feature", TORCH_FN(facebook::recir::kjtSplitPerFeature));
  m.impl(
      "kjt_lengths_per_feature",
      TORCH_FN(facebook::recir::kjtLengthsPerFeature));
}

TORCH_LIBRARY_FRAGMENT(generative_recommenders, m) {
  m.def(
      torch::schema(
          "concat_2D_jagged_fwd_op(int max_seq_len, Tensor values_a, Tensor values_b, Tensor? offsets_a, Tensor? offsets_b, bool is_replace, int n_prefix_from_right) -> Tensor",
          c10::AliasAnalysisKind::PURE_FUNCTION));
  m.def(
      torch::schema(
          "triton_weighted_layer_norm_fwd(Tensor x, Tensor? weight, Tensor? bias, float eps, Tensor? mean=None, Tensor? rstd=None, bool is_swish=False) -> (Tensor, Tensor, Tensor)",
          c10::AliasAnalysisKind::PURE_FUNCTION));
  m.def(
      torch::schema(
          "triton_addmm_fwd(Tensor x, Tensor w, Tensor y) -> Tensor",
          c10::AliasAnalysisKind::PURE_FUNCTION));
  m.def(
      torch::schema(
          "triton_jagged_dense_bmm_add_fwd(int max_seq_len, Tensor seq_offsets, Tensor jagged, Tensor dense, Tensor bias, bool elementwise=False) -> (Tensor, int, int, int)",
          c10::AliasAnalysisKind::PURE_FUNCTION));
  m.def(
      torch::schema(
          "split_2D_jagged_fwd_op(Tensor values, int max_seq_len, Tensor offsets_a, Tensor offsets_b, int seq_len_a, int seq_len_b, bool is_dense_a, bool is_dense_b, int dense_size, int n_prefix_to_right, int B) -> (Tensor, Tensor)",
          c10::AliasAnalysisKind::PURE_FUNCTION));
  m.def(
      torch::schema(
          "_triton_weighted_layer_norm_quantization_fwd_impl(Tensor x, Tensor weight, Tensor bias, float eps, Tensor mean, Tensor rstd, int BLOCK_D) -> (Tensor, Tensor, Tensor, Tensor, Tensor)",
          c10::AliasAnalysisKind::PURE_FUNCTION));
  m.def(
      torch::schema(
          "_triton_layer_norm_mul_dropout_quantization_fwd_impl(Tensor x, Tensor u, Tensor weight, Tensor bias, float eps, float dropout_ratio, bool training, bool silu_u, bool concat_u, bool concat_x, str mul_u_activation_type, int seed, bool save_y, int BLOCK_D) -> (Tensor, Tensor, Tensor, Tensor, Tensor)",
          c10::AliasAnalysisKind::PURE_FUNCTION));
  m.def(
      torch::schema(
          "_triton_fp8_rowwise_addmm_fwd_impl(Tensor a, Tensor b, Tensor a_scale, Tensor b_scale, Tensor bias, bool has_bias, bool bias_is_2d, ScalarType dot_out_dtype, bool allow_tf32, bool use_fast_accum) -> Tensor",
          c10::AliasAnalysisKind::PURE_FUNCTION));
}

TORCH_LIBRARY_IMPL(generative_recommenders, CompositeImplicitAutograd, m) {
  m.impl("concat_2D_jagged_fwd_op", TORCH_FN(facebook::recir::concat2DJagged));
  m.impl(
      "triton_weighted_layer_norm_fwd",
      TORCH_FN(facebook::recir::weightedLayerNorm));
  m.impl("triton_addmm_fwd", TORCH_FN(facebook::recir::tritonAddMm));
  m.impl(
      "triton_jagged_dense_bmm_add_fwd",
      TORCH_FN(facebook::recir::jaggedDenseBmmAdd));
  m.impl("split_2D_jagged_fwd_op", TORCH_FN(facebook::recir::split2DJagged));
  m.impl(
      "_triton_weighted_layer_norm_quantization_fwd_impl",
      TORCH_FN(facebook::recir::weightedLayerNormQuantized));
  m.impl(
      "_triton_layer_norm_mul_dropout_quantization_fwd_impl",
      TORCH_FN(facebook::recir::layerNormMulQuantized));
  m.impl(
      "_triton_fp8_rowwise_addmm_fwd_impl",
      TORCH_FN(facebook::recir::fp8RowwiseAddMm));
}

TORCH_LIBRARY_FRAGMENT(hammer, m) {
  m.def(
      torch::schema(
          "add_ts_pos_emb_fwd_op(Tensor seq_embeddings, Tensor seq_offsets, Tensor pos_embeddings, Tensor ts_embeddings, Tensor timestamps, Tensor seq_lengths, Tensor? num_targets, int max_seq_len, int max_contextual_seq_len, bool interleave_targets, str time_bucket_fn) -> (Tensor, Tensor, Tensor, Tensor, Tensor)",
          c10::AliasAnalysisKind::PURE_FUNCTION));
  m.def(
      torch::schema(
          "triton_ragged_attention_fwd(int N, float alpha, Tensor q, Tensor k, Tensor v, Tensor seq_offsets, str invalid_attn_mask_type, Tensor? num_targets, Tensor? attn_scale, int max_attn_len, int contextual_seq_len, int full_attn_size, int num_softmax_heads=0) -> (Tensor, Tensor)",
          c10::AliasAnalysisKind::PURE_FUNCTION));
}

TORCH_LIBRARY_IMPL(hammer, CompositeImplicitAutograd, m) {
  m.impl(
      "add_ts_pos_emb_fwd_op",
      TORCH_FN(facebook::recir::addTimestampPositionEmbeddings));
  m.impl(
      "triton_ragged_attention_fwd",
      TORCH_FN(facebook::recir::raggedAttention));
}
