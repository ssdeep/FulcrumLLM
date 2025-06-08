package com.ssdeep.fulcrum.transformer

import com.ssdeep.fulcrum.transformer.config.GPTConfig
import com.ssdeep.fulcrum.transformer.normalization.LayerNormalization
import torch.{Device, Float32, Int64, Tensor}

case class GPTModel(cfg: GPTConfig) extends torch.nn.Module:
  private val tokEmbed = register(torch.nn.Embedding(cfg.vocabSize.intValue(), cfg.embedDim), "tokEmbed")
  private val posEmbed = register(torch.nn.Embedding(cfg.contextLength, cfg.embedDim), "posEmbed")
  private val dropEmbed = register(torch.nn.Dropout(cfg.dropRate), "dropEmbed")

  private val trfBlocks = register(torch.nn.Sequential(
    (0 until cfg.numLayers).map(
      _ => TransformerBlock(cfg)
    ):_*
  ), "trfBlocks")

  private val finalNorm = register(LayerNormalization(cfg.embedDim), "finalNorm")
  private val outHead = register(torch.nn.Linear(
    cfg.embedDim,
    cfg.vocabSize,
    addBias = false
  ), "outHead")

  def totalParams: Long = this.parameters(true).map(_.numel).sum
  def printSizeInMB: Unit = {
    val totalSize = (totalParams*4.0)/(1024.0*1024)
    println(f"Total Model Size: $totalSize%.2f MB")
  }

  val posEmbedsMap: scala.collection.mutable.Map[Int, Tensor[Float32]] = scala.collection.mutable.Map()

  def apply(inIdx: Tensor[Int64]): Tensor[Float32] =
    val Seq(batchSize, seqLen) = inIdx.shape

    val tokEmbeds = tokEmbed(inIdx)
    val posEmbeds = {
      posEmbedsMap.getOrElseUpdate(
        seqLen,
        posEmbed(torch.arange(end = seqLen, device = inIdx.device, dtype = torch.int64))
      )
    }
    var x = tokEmbeds + posEmbeds
    x = dropEmbed(x)
    x = trfBlocks(x)
    x = finalNorm(x)
    val logits = outHead(x)
    x.native.deallocate()
    logits



