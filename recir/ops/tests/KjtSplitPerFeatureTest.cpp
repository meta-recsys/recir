// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "recir/ops/kjt_split_per_feature.h"

#include <ATen/ATen.h>
#include <gtest/gtest.h>

namespace facebook::recir {
namespace {

TEST(KjtSplitPerFeatureTest, SplitsValuesAtPerFeatureOffsets) {
  const auto options = at::TensorOptions().dtype(at::kLong);
  const auto values = at::arange(12, options);
  const auto offsets = at::tensor({0, 2, 5, 5, 10, 11, 12}, options);

  const auto result = kjtSplitPerFeature(values, offsets, 2, 3);

  ASSERT_EQ(result.size(), 3);
  EXPECT_TRUE(result[0].equal(at::tensor({0, 1, 2, 3, 4}, options)));
  EXPECT_TRUE(result[1].equal(at::tensor({5, 6, 7, 8, 9}, options)));
  EXPECT_TRUE(result[2].equal(at::tensor({10, 11}, options)));
}

TEST(KjtSplitPerFeatureTest, RejectsMissingTerminalOffset) {
  const auto options = at::TensorOptions().dtype(at::kLong);
  const auto values = at::arange(4, options);
  const auto offsets = at::tensor({0, 2, 4}, options);

  EXPECT_THROW(kjtSplitPerFeature(values, offsets, 2, 2), c10::Error);
}

TEST(KjtLengthsPerFeatureTest, DerivesStrideFromRuntimeOffsets) {
  const auto options = at::TensorOptions().dtype(at::kLong);
  const auto offsets = at::tensor({0, 2, 5}, options);

  const auto result = kjtLengthsPerFeature(offsets, 2);

  ASSERT_EQ(result.size(), 2);
  EXPECT_TRUE(result[0].equal(at::tensor({2}, options)));
  EXPECT_TRUE(result[1].equal(at::tensor({3}, options)));
}

TEST(KjtLengthsPerFeatureTest, RejectsUnevenRuntimeStride) {
  const auto options = at::TensorOptions().dtype(at::kLong);
  const auto offsets = at::tensor({0, 1, 2, 3}, options);

  EXPECT_THROW(kjtLengthsPerFeature(offsets, 2), c10::Error);
}

TEST(Concat2DJaggedTest, ConcatenatesJaggedValuesWithRightPrefix) {
  const auto options = at::TensorOptions().dtype(at::kLong);
  const auto valuesA = at::tensor({10, 11, 20}, options).view({3, 1});
  const auto valuesB = at::tensor({1, 2, 3, 4, 5}, options).view({5, 1});
  const auto offsetsA = at::tensor({0, 2, 3}, options);
  const auto offsetsB = at::tensor({0, 3, 5}, options);

  const auto result =
      concat2DJagged(5, valuesA, valuesB, offsetsA, offsetsB, false, 1);

  const auto expected =
      at::tensor({1, 10, 11, 2, 3, 4, 20, 5}, options).view({8, 1});
  EXPECT_TRUE(result.equal(expected));
}

TEST(Concat2DJaggedTest, ReplacesJaggedSuffix) {
  const auto options = at::TensorOptions().dtype(at::kLong);
  const auto valuesA = at::tensor({10, 11, 12, 20, 21}, options).view({5, 1});
  const auto valuesB = at::tensor({1, 2, 3}, options).view({3, 1});
  const auto offsetsA = at::tensor({0, 3, 5}, options);
  const auto offsetsB = at::tensor({0, 1, 3}, options);

  const auto result =
      concat2DJagged(3, valuesA, valuesB, offsetsA, offsetsB, true, 0);

  const auto expected = at::tensor({10, 11, 1, 2, 3}, options).view({5, 1});
  EXPECT_TRUE(result.equal(expected));
}

TEST(Concat2DJaggedTest, SupportsDenseLeftValues) {
  const auto options = at::TensorOptions().dtype(at::kLong);
  const auto valuesA = at::tensor({10, 11, 20, 21}, options).view({2, 2, 1});
  const auto valuesB = at::tensor({1, 2, 3}, options).view({3, 1});
  const auto offsetsB = at::tensor({0, 1, 3}, options);

  const auto result =
      concat2DJagged(4, valuesA, valuesB, std::nullopt, offsetsB, false, 0);

  const auto expected =
      at::tensor({10, 11, 1, 20, 21, 2, 3}, options).view({7, 1});
  EXPECT_TRUE(result.equal(expected));
}

TEST(WeightedLayerNormTest, AppliesLearnableNormalization) {
  const auto options = at::TensorOptions().dtype(at::kFloat);
  const auto x = at::tensor({1.0, 3.0, 2.0, 6.0}, options).view({2, 2});
  const auto weight = at::tensor({2.0, 3.0}, options);
  const auto bias = at::tensor({1.0, -1.0}, options);

  const auto [result, mean, reciprocalStd] = weightedLayerNorm(
      x, weight, bias, 0.0, std::nullopt, std::nullopt, false);

  EXPECT_TRUE(
      result.equal(at::tensor({-1.0, 2.0, -1.0, 2.0}, options).view({2, 2})));
  EXPECT_TRUE(mean.equal(at::tensor({2.0, 4.0}, options)));
  EXPECT_TRUE(reciprocalStd.equal(at::tensor({1.0, 0.5}, options)));
}

TEST(TritonAddMmTest, AddsBroadcastBias) {
  const auto options = at::TensorOptions().dtype(at::kFloat);
  const auto x = at::tensor({1.0, 2.0, 3.0, 4.0}, options).view({2, 2});
  const auto weight = at::tensor({1.0, 2.0, 3.0, 4.0}, options).view({2, 2});
  const auto bias = at::tensor({10.0, 20.0}, options);

  const auto result = tritonAddMm(x, weight, bias);

  EXPECT_TRUE(
      result.equal(at::tensor({17.0, 30.0, 25.0, 42.0}, options).view({2, 2})));
}

TEST(JaggedDenseBmmAddTest, MultipliesEachJaggedBatchAndAddsBias) {
  const auto options = at::TensorOptions().dtype(at::kFloat);
  const auto offsets =
      at::tensor({0, 2, 3}, at::TensorOptions().dtype(at::kLong));
  const auto jagged =
      at::tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, options).view({3, 2});
  const auto dense =
      at::tensor({1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0}, options)
          .view({2, 2, 2});
  const auto bias = at::tensor({10.0, 20.0, 30.0, 40.0}, options).view({2, 2});

  const auto [result, batchSize, inputDimension, outputDimension] =
      jaggedDenseBmmAdd(2, offsets, jagged, dense, bias, false);

  EXPECT_TRUE(result.equal(
      at::tensor({11.0, 22.0, 13.0, 24.0, 40.0, 58.0}, options).view({3, 2})));
  EXPECT_EQ(batchSize, 2);
  EXPECT_EQ(inputDimension, 2);
  EXPECT_EQ(outputDimension, 2);
}

TEST(AddTimestampPositionEmbeddingsTest, AddsPositionAndTimestampBuckets) {
  const auto floatOptions = at::TensorOptions().dtype(at::kFloat);
  const auto longOptions = at::TensorOptions().dtype(at::kLong);
  const auto sequenceEmbeddings = at::zeros({3, 1}, floatOptions);
  const auto sequenceOffsets = at::tensor({0, 3}, longOptions);
  const auto positionEmbeddings =
      at::tensor({0.0, 10.0, 20.0, 30.0, 40.0}, floatOptions).view({5, 1});
  const auto timestampEmbeddings =
      at::arange(2049, floatOptions).unsqueeze(1) * 100.0;
  const auto timestamps =
      at::tensor({10.0, 40.0, 70.0}, floatOptions).view({1, 3});
  const auto sequenceLengths = at::tensor({3}, longOptions);

  const auto [result, sortedPositions, positionOrder, sortedTimes, timeOrder] =
      addTimestampPositionEmbeddings(
          sequenceEmbeddings,
          sequenceOffsets,
          positionEmbeddings,
          timestampEmbeddings,
          timestamps,
          sequenceLengths,
          std::nullopt,
          3,
          1,
          false,
          "sqrt");

  EXPECT_TRUE(
      result.equal(at::tensor({100.0, 30.0, 20.0}, floatOptions).view({3, 1})));
  const auto intOptions = at::TensorOptions().dtype(at::kInt);
  EXPECT_TRUE(sortedPositions.equal(at::tensor({0, 2, 3}, intOptions)));
  EXPECT_TRUE(sortedTimes.equal(at::tensor({0, 0, 1}, intOptions)));
  EXPECT_EQ(positionOrder.numel(), 3);
  EXPECT_EQ(timeOrder.numel(), 3);
}

TEST(Split2DJaggedTest, ReversesConcatenationWithRightPrefix) {
  const auto options = at::TensorOptions().dtype(at::kLong);
  const auto values =
      at::tensor({1, 10, 11, 2, 3, 4, 20, 5}, options).view({8, 1});
  const auto offsetsA = at::tensor({0, 2, 3}, options);
  const auto offsetsB = at::tensor({0, 3, 5}, options);

  const auto [valuesA, valuesB] =
      split2DJagged(values, 5, offsetsA, offsetsB, 3, 5, false, false, 0, 1, 2);

  EXPECT_TRUE(valuesA.equal(at::tensor({10, 11, 20}, options).view({3, 1})));
  EXPECT_TRUE(valuesB.equal(at::tensor({1, 2, 3, 4, 5}, options).view({5, 1})));
}

TEST(Fp8RowwiseAddMmTest, DequantizesRowsBeforeMatrixMultiplication) {
  const auto options = at::TensorOptions().dtype(at::kFloat);
  const auto a = at::tensor({1.0, 2.0}, options).view({1, 2});
  const auto b = at::tensor({3.0, 4.0, 5.0, 6.0}, options).view({2, 2});
  const auto aScale = at::tensor({2.0}, options);
  const auto bScale = at::tensor({3.0, 4.0}, options);
  const auto bias = at::tensor({1.0, 2.0}, options);

  const auto result = fp8RowwiseAddMm(
      a, b, aScale, bScale, bias, true, false, at::kFloat, true, true);

  EXPECT_TRUE(result.equal(at::tensor({67.0, 138.0}, options).view({1, 2})));
}

TEST(LayerNormMulQuantizedTest, ConcatenatesInputsAndNormalizedProduct) {
  const auto options = at::TensorOptions().dtype(at::kFloat);
  const auto x = at::tensor({1.0, 3.0}, options).view({1, 2});
  const auto u = at::tensor({2.0, 4.0}, options).view({1, 2});
  const auto weight = at::ones({2}, options);
  const auto bias = at::zeros({2}, options);

  const auto [output, mean, reciprocalStd, scale, quantized] =
      layerNormMulQuantized(
          x,
          u,
          weight,
          bias,
          0.0,
          0.0,
          false,
          false,
          true,
          true,
          "none",
          0,
          true,
          2);

  EXPECT_TRUE(output.equal(
      at::tensor({2.0, 4.0, 1.0, 3.0, -2.0, 4.0}, options).view({1, 6})));
  EXPECT_TRUE(mean.equal(at::tensor({2.0}, options)));
  EXPECT_TRUE(reciprocalStd.equal(at::tensor({1.0}, options)));
  EXPECT_EQ(scale.numel(), 1);
  EXPECT_EQ(quantized.sizes(), output.sizes());
}

TEST(RaggedAttentionTest, ReturnsJaggedOutputForSiluAttention) {
  const auto options = at::TensorOptions().dtype(at::kFloat);
  const auto query = at::zeros({2, 1, 1}, options);
  const auto key = at::zeros({2, 1, 1}, options);
  const auto value = at::tensor({2.0, 3.0}, options).view({2, 1, 1});
  const auto offsets = at::tensor({0, 2}, at::TensorOptions().dtype(at::kLong));

  const auto [output, auxiliary] = raggedAttention(
      2,
      1.0,
      query,
      key,
      value,
      offsets,
      "lower_triangular",
      std::nullopt,
      std::nullopt,
      0,
      0,
      0,
      0);

  EXPECT_TRUE(output.equal(at::zeros_like(value)));
  EXPECT_EQ(auxiliary.numel(), 1);
}

} // namespace
} // namespace facebook::recir
