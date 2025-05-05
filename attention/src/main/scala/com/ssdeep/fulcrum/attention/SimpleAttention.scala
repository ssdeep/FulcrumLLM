package com.ssdeep.fulcrum.attention

import com.ssdeep.fulcrum.embedding.GPTDatasetV1
import torch.{Float32, Int64, Tensor}
import org.bytedeco.pytorch.global.{torch => pytorch}

class SimpleAttention(dataset: GPTDatasetV1):
  def processBatch(batch: Tensor[Float32]) =
    val batchTensor = (0 until batch.shape.head).map {
        i =>
          val candidate = batch(i)
          println("Candidate Tensor")
          println(candidate)
          (0 until batch.shape.head).map {
            targetIndex =>
              val target = batch(targetIndex)
              println("Target Tensor")
              println(target)
              val dotProduct = pytorch.dot(candidate.native, target.native)
              dotProduct.item_float()
          }
      }
    torch.Tensor.apply(batchTensor)

  def getAttention =
    dataset.dataLoader
      .map {
        case (inputBatch, targetBatch) =>
               println(s"Input shape${inputBatch.shape}")
               println(s"Input row: ${inputBatch(0)}")
               val inputEmbeddings = dataset.inputEmbeddings(inputBatch)
               val processedBatch = processBatch(inputEmbeddings(0))
               println(s"$processedBatch")
      }

