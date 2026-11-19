// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <ATen/core/dispatch/Dispatcher.h>
#include <gtest/gtest.h>

extern "C" void recir_kjt_split_per_feature_registration_anchor();

namespace facebook::recir {
namespace {

TEST(RecirOpsSharedLibraryTest, RegistersOperatorSchemasWhenLoaded) {
  recir_kjt_split_per_feature_registration_anchor();

  EXPECT_NO_THROW(
      c10::Dispatcher::singleton().findSchemaOrThrow(
          "recir_torchrec::kjt_split_per_feature", ""));
  EXPECT_NO_THROW(
      c10::Dispatcher::singleton().findSchemaOrThrow(
          "generative_recommenders::concat_2D_jagged_fwd_op", ""));
  EXPECT_NO_THROW(
      c10::Dispatcher::singleton().findSchemaOrThrow(
          "hammer::triton_ragged_attention_fwd", ""));
}

} // namespace
} // namespace facebook::recir
