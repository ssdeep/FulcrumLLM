package com.ssdeep.fulcrum.attention

import com.ssdeep.fulcrum.attention.parameters.ParameterModule
import com.ssdeep.fulcrum.embedding.GPTDatasetV1
import org.bytedeco.pytorch.global.torch as pytorch

class WeightedAttention(dataset: GPTDatasetV1):
  def getWeightedAttention =
        dataset.dataLoader
          .map {
            case (inputBatch, targetBatch) =>
            println(s"Input shape${inputBatch.shape}")
            println(s"Input row: ${inputBatch(0)}")
            val inputEmbeddingsBatch = dataset.inputEmbeddings(inputBatch)
            val input1 = inputEmbeddingsBatch(1)
            val x2 = input1(1)
            val dIn = input1.shape(1)
            val dOut = 100
            torch.manualSeed(123)
            val params = ParameterModule(dIn, dOut)
            val tests = params.processInput(x2)
            val keys = inputEmbeddingsBatch `@` params.weightK
            val values = inputEmbeddingsBatch `@` params.weightV
            println("All Keys")
            println(keys)
            println("All Values")
            println(values)
            println(tests)
            val queries = inputEmbeddingsBatch.matmul(params.weightQ)
            val attScores = queries.dot2d(keys)
            val attWeights = torch.softmax(attScores/Math.pow(keys.shape.last, 0.5), dim = 0L, dtype = torch.float32)
            println(s"Attention Scores: $queries")
            val contextVectors = attWeights `@` values
            println("Context vectors")
            println(contextVectors)
            contextVectors
        }

