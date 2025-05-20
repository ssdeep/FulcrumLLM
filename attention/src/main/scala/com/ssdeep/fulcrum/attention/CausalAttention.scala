package com.ssdeep.fulcrum.attention

import torch.{Float32, Tensor, nn}
import org.bytedeco.pytorch.global.torch as pytorch
import torch.Tensor.fromNative
import scala.jdk.CollectionConverters._

class CausalAttention(
                       dIn: Int,
                       dOut: Int,
                       contextLength: Int,
                       dropoutFactor: Double = 0.5,
                       qkvBias: Boolean = false) extends torch.nn.Module:
  val Wquery = nn.Linear(dIn, dOut, qkvBias)
  val Wkey = nn.Linear(dIn, dOut, qkvBias)
  val Wvalue = nn.Linear(dIn, dOut, qkvBias)
  val dropout = nn.Dropout(p = dropoutFactor)

  registerBuffer(
    name = "mask",
    tensor = fromNative(pytorch.triu(pytorch.ones(contextLength, contextLength), 1))
  )

  def forward(inputBatch: Tensor[Float32]): Tensor[Float32] =
    val keys = Wkey(inputBatch)
    val values = Wvalue(inputBatch)
    val queries = Wquery.apply(inputBatch)

    val Seq(batchSize, numTokens, dIn) = inputBatch.shape
    var attentionScores: Tensor[Float32] = queries `@` keys.transpose(1,2)
    val bufferedMask = this.namedBuffers(true)("mask").to(torch.bool)
    attentionScores = attentionScores.maskedFill(
      bufferedMask,
      Float.NegativeInfinity
    )
    val attnWeights = torch.softmax(attentionScores/Math.pow(keys.shape.last, 0.5), dim = -1L, dtype = torch.float32)
    val attnWeightDropouts: Tensor[Float32] = dropout(attnWeights)
//    val attnWeightsMasked = attnWeightDropouts * bufferedMask
//    val rowSums = attnWeightsMasked.sum(dim = -1, keepdim = true, dtype = torch.float32)
    val contextVectors = attnWeightDropouts `@` values
//    val reverseContext = attnWeightDropouts `@` values
    contextVectors




