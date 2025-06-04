package com.ssdeep.fulcrum.attention

import com.ssdeep.fulcrum.embedding.GPTDatasetV1
import com.ssdeep.fulcrum.tokenizer.Tokenizer
import org.scalacheck.Gen
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks

class MultiheadAttentionSuite extends AnyFlatSpec with ScalaCheckPropertyChecks {
  val tokenizer = Tokenizer.buildDefaultO200kBase
  val generatableText = (i: Int) => Gen.listOfN(i, Gen.asciiChar).flatMap(p => p.mkString(""))
  "MultiheadAttention Context Vectors" should "satisfy dimensionality" in {
    forAll(generatableText(100)) {
       text =>
         val MAX_LENGTH = 5L
         val BATCH_SIZE = 8
         val gptDataset = GPTDatasetV1(
           text,
           tokenizer = tokenizer,
           maxLength = MAX_LENGTH.intValue(),
           stride = 1,
           batchSize = BATCH_SIZE
         )
         torch.manualSeed(128L)
         var datasetIterator  = gptDataset.dataLoader.iterator.iterator
         var (batch, _) = datasetIterator.next()
         var batchEmbeddings = gptDataset.inputEmbeddings(batch)
         val multiHeadAttn = MultiheadAttention(
           batchEmbeddings.shape.last,
           batchEmbeddings.shape.last,
           contextLength = MAX_LENGTH.intValue(),
           dropoutFactor = 0.1,
           numHeads = 8
         )
         var multiHeadContextVector = multiHeadAttn.apply(batchEmbeddings)
         while {
           multiHeadContextVector.shape.length === 2
           batch = datasetIterator.next()._1
           multiHeadContextVector = multiHeadAttn.apply(gptDataset.inputEmbeddings(batch))
           datasetIterator.hasNext
         } do()
    }
  }
}
