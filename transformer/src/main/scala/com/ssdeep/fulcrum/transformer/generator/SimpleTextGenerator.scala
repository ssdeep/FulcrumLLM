package com.ssdeep.fulcrum.transformer.generator

import com.ssdeep.fulcrum.transformer.GPTModel
import com.ssdeep.fulcrum.attention.span
import com.ssdeep.fulcrum.core.ClassNameLogger
import torch.{DType, Float32, Int64, Tensor, noGrad}

case class SimpleTextGenerator(model: GPTModel) extends ClassNameLogger:

  def generateSimpleText(idx: Tensor[Int64], maxNewTokens: Int, contextLength: Int): Tensor[Int64] =
    var idxTemp = idx
    val maxCols = idxTemp.shape(1)
    (0 until Math.min(maxNewTokens, maxCols)).foreach {
      i =>
        val idxCond = idxTemp.span(None, Some(-contextLength.longValue()))

        var logits = noGrad {
         model(idxCond)
        }

        logits = torch.select(logits, 1, -1) // equivalent to logits[:,-1,:]
        val probas = torch.softmax(logits, dim = -1L, dtype = torch.float32)
        val idxNext = torch.argmax(probas, dim = -1, keepdim = true)
        idxTemp = torch.cat(Seq(idxTemp, idxNext), dim = -1)
        logger.info(s"Tokens after round $i")
        logger.info(idxTemp)
    }
    idxTemp

