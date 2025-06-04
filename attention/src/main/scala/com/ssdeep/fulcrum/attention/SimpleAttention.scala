package com.ssdeep.fulcrum.attention

import com.ssdeep.fulcrum.embedding.GPTDatasetV1
import torch.{Float32, Int64, Tensor}
import org.bytedeco.pytorch.global.torch as pytorch

class SimpleAttention(dataset: GPTDatasetV1):
  //TODO: Move this to a utility
  def getContextVector(tensor1: Tensor[Float32], tensor2: Tensor[Float32]) =
    val ti = (0 until tensor1.shape.head).map {
      i =>
        val tensor1i = tensor1(i)
        val contextVector = (0 until tensor2.shape.head).map {
            j =>
              val contextij = tensor1i(j) * tensor2(j)
              contextij.toSeq
          }
        println(s"New context vector $contextVector")
        contextVector
    }
    torch.Tensor.apply(ti)

//    val dotproducts = (0 until tensor1.shape.head).map {
//      i =>
//        tensor1(i)*tensor2(i)
////        pytorch.dot(tensor1(i).native, tensor2(i).native).item_float()
//    }
//    torch.Tensor(dotproducts)


  def processBatch(batch: Tensor[Float32]) =
    val batchTensor = (0 until batch.shape.head).map {
        i =>
          val candidate = batch(i)
          println("Candidate Tensor")
          println(candidate)
          val rowTensor = (0 until batch.shape.head).map {
            targetIndex =>
              val target = batch(targetIndex)
              println("Target Tensor")
              println(target)
              val dotProduct = pytorch.dot(candidate.native, target.native)
              dotProduct.item_float()
          }
          rowTensor.map(_/rowTensor.sum)
      }.toList
    val rawResultTensor = torch.Tensor.apply(batchTensor)
    val normalzied = torch.softmax(rawResultTensor, dim = 0L, dtype = torch.float32)
    println("Post normalization")
    println(normalzied)
    normalzied

  def getAttention =
    dataset.dataLoader
      .map {
        case (inputBatch, targetBatch) =>
               println(s"Input shape${inputBatch.shape}")
               println(s"Input row: ${inputBatch(0)}")
               val inputEmbeddings = dataset.inputEmbeddings(inputBatch)
               (0 until inputEmbeddings.shape.head).map {
                 i =>
                   val processedBatch = processBatch(inputEmbeddings(i))
                   println(s"Input embedding ${inputEmbeddings(i)}")
                   println(s"Processed batch: $processedBatch")
                   val contextVector = getContextVector(processedBatch, inputEmbeddings(i))
                   println(s"dot product: $contextVector")
                   contextVector
               }

      }


