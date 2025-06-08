package com.ssdeep.fulcrum.embedding

import com.ssdeep.fulcrum.tokenizer.Tokenizer
import torch.*
import torch.data.TensorDataset

import scala.collection.mutable.ArrayBuffer

/**
 * Reference Dataset that takes text and returns input and target tensors
 */
object GPTDatasetV1:
  def apply(text: String, tokenizer: Tokenizer, maxLength: Int, stride: Int, batchSize: Int, dimensionality: Int = 256,
            shuffle: Boolean = false)
  : GPTDatasetV1 =
    val tokenIds: List[Long] = tokenizer.encode(text)
    GPTDatasetV1(tokenIds, tokenizer, maxLength, stride, batchSize, dimensionality, shuffle)


/**
 * Reference Dataset that takes tokenIds and returns input and target tensors
 */
case class GPTDatasetV1(tokenIds: List[Long], tokenizer: Tokenizer, maxLength: Int, stride: Int, batchSize: Int,
                        dimensionality: Int, shuffle: Boolean):
  private val extraTokensToDrop = tokenIds.length % maxLength // if the context length doesn't evenly divide the input
  private val rawBatchedTokens = tokenIds.dropRight(extraTokensToDrop).sliding(maxLength, stride).toSeq
  private val inputChunks = rawBatchedTokens.dropRight(1).iterator.grouped(batchSize).map(Tensor(_))
  private val targetChunks = rawBatchedTokens.drop(1).iterator.grouped(batchSize).map(Tensor(_))
  private val totalChunks = rawBatchedTokens.length / batchSize

  private lazy val tokenEmbeddingLayer = new torch.nn.Embedding(tokenizer.tokenIdMap.size, dimensionality)
  private lazy val posEmbeddingLayer = new torch.nn.Embedding(maxLength, dimensionality)
  def ~>(inputs: Tensor[Int64]): Tensor[Float32] = tokenEmbeddingLayer.apply(inputs)
  def pos(inputs: Tensor[Int64]): Tensor[Float32] = posEmbeddingLayer.apply(inputs)
  def inputEmbeddings(inputs: Tensor[Int64]): Tensor[Promoted[Float32, Float32]] = ~>(inputs) + pos(torch.arange(end = 
    maxLength.longValue()))
  private lazy val dataset = inputChunks.zip(targetChunks)
  lazy val dataLoaderAndCount = (dataset, totalChunks)
  lazy val dataLoader = dataset