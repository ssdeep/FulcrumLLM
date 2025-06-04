package com.ssdeep.fulcrum.transformer.config

import com.typesafe.config.Config

case class GPTConfig(
                      vocabSize: Long,
                      contextLength: Int,
                      embedDim: Int,
                      numHeads: Int,
                      numLayers: Int,
                      dropRate: Double,
                      qkvBias: Boolean)

object GPTConfig:
  def loadFromConfig(config: Config): GPTConfig = {
    val gptConfig = config.getConfig("gpt-config")
    GPTConfig(
      gptConfig.getLong("vocab_size"),
      gptConfig.getInt("context_length"),
      gptConfig.getInt("emd_dim"),
      gptConfig.getInt("n_heads"),
      gptConfig.getInt("n_layers"),
      gptConfig.getDouble("drop_rate"),
      gptConfig.getBoolean("qkv_bias")
    )
  }