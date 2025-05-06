package com.ssdeep.fulcrum.tokenizer

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks


class TokenizerSuite extends AnyFlatSpec with ScalaCheckPropertyChecks  {
  val tokenizer: Tokenizer = Tokenizer.buildDefaultO200kBase

  "String" should "encode and decode without loss" in {
     forAll {
       (text: String) =>
         println(s"Running with $text")
         tokenizer.decode(tokenizer.encode(text)) === text
     }
  }

}
