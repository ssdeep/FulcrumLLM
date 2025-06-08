package com.ssdeep.fulcrum.transformer.generator

import com.ssdeep.fulcrum.transformer.GPTModel
import com.ssdeep.fulcrum.attention.span
import torch.{DType, Float32, Int64, Tensor, noGrad}

case class SimpleTextGenerator(model: GPTModel):

  def generateSimpleText(idx: Tensor[Int64], maxNewTokens: Int, contextLength: Int): Tensor[Int64] =
    var idxTemp = idx
    (0 until maxNewTokens).foreach {
      i =>
        val idxCond = idxTemp.span(None, Some(-contextLength.longValue()))

        var logits = noGrad {
         model(idxCond)
        }

        logits = torch.select(logits, 1, -1) // equivalent to logits[:,-1,:]
        val probas = torch.softmax(logits, dim = -1L, dtype = torch.float32)
        val idxNext = torch.argmax(probas, dim = -1, keepdim = true)
        idxTemp = torch.cat(Seq(idxTemp, idxNext), dim = -1)
        println(s"Tokens after round $i")
        println(idxTemp)
    }
    idxTemp

