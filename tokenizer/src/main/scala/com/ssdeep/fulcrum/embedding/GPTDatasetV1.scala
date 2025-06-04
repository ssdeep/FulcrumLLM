package com.ssdeep.fulcrum.embedding

import com.ssdeep.fulcrum.tokenizer.Tokenizer
import torch.*
import torch.data.TensorDataset

import scala.collection.mutable.ArrayBuffer

/**
 * Reference Dataset that takes text and returns input and target tensors
 */
case class GPTDatasetV1(text: String, tokenizer: Tokenizer, maxLength: Int, stride: Int, batchSize: Int, dimensionality: Int =
256):

  private val tokenIds: List[Long] = tokenizer.encode(text)
  private val inputChunks = torch.Tensor(tokenIds.sliding(maxLength, stride).toSeq.dropRight(1))
  private val targetChunks = torch.Tensor(tokenIds.drop(1).sliding(maxLength, stride).toSeq)

  private lazy val tokenEmbeddingLayer = new torch.nn.Embedding(tokenizer.tokenIdMap.size, dimensionality)
  private lazy val posEmbeddingLayer = new torch.nn.Embedding(maxLength, dimensionality)
  def ~>(inputs: Tensor[Int64]): Tensor[Float32] = tokenEmbeddingLayer.apply(inputs)
  def pos(inputs: Tensor[Int64]): Tensor[Float32] = posEmbeddingLayer.apply(inputs)
  def inputEmbeddings(inputs: Tensor[Int64]): Tensor[Promoted[Float32, Float32]] = ~>(inputs) + pos(torch.arange(end = 
    maxLength.longValue()))
  private lazy val dataset = torch.data.TensorDataset(inputChunks, targetChunks)
  lazy val dataLoader = torch.data.TupleDataLoader(
    dataset,
    batchSize = batchSize,
    shuffle = false
  )