package com.ssdeep.fulcrum.attention

import org.bytedeco.pytorch.global.torch as pytorch
import torch.Tensor.fromNative
import torch.{Float32, Tensor, nn}

class MultiheadAttention(
                          dIn: Int,
                          dOut: Int,
                          contextLength: Int,
                          dropoutFactor: Double = 0.5,
                          qkvBias: Boolean = false,
                          numHeads: Int = 5
                        ) extends torch.nn.Module:
  require(dOut % numHeads == 0, s"output dimension $dOut must be divisible by $numHeads")
  val headDim: Int = dOut/numHeads
  val Wquery = register(nn.Linear(dIn, dOut, qkvBias))
  val Wkey = register(nn.Linear(dIn, dOut, qkvBias))
  val Wvalue = register(nn.Linear(dIn, dOut, qkvBias))
  val dropout = register(nn.Dropout(p = dropoutFactor))
  val outProjection = register(nn.Linear(dOut, dOut))
  registerBuffer(
    "mask",
    tensor = fromNative(pytorch.triu(pytorch.ones(contextLength, contextLength), 1))
  )
  
  def apply(input: Tensor[Float32]): Tensor[Float32] =
    val Seq(batch, numTokens, inputLength) = input.shape
    val keys = Wkey(input).view(batch, numTokens, numHeads, headDim).transpose(1, 2)
    val queries = Wquery(input).view(batch, numTokens, numHeads, headDim).transpose(1, 2)
    val values = Wvalue(input).view(batch, numTokens, numHeads, headDim).transpose(1, 2)
    
    var attnScores = queries `@` keys.transpose(2, 3)
    val bufferedMask = this.namedBuffers(true)("mask").to(torch.bool).span(Some(numTokens), Some(numTokens))
    attnScores = attnScores.maskedFill(
      bufferedMask,
      Float.NegativeInfinity
    )
    val attnWeights =  torch.softmax(attnScores/Math.pow(keys.shape.last, 0.5), dim = -1L, dtype = torch.float32)
    val attnWeightDropouts: Tensor[Float32] = dropout(attnWeights)
    
    val contextVector = (attnWeightDropouts `@` values)
      .transpose(1, 2)
      .contiguous
      .view(batch, numTokens, dOut)
    outProjection(contextVector)
    
