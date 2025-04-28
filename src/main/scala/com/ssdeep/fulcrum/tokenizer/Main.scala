package com.ssdeep.fulcrum.tokenizer

import org.json4s.*
import org.json4s.native.JsonMethods.*
import org.json4s.native.JsonMethods
import org.json4s.JsonDSL._

import java.io.{FileOutputStream, PrintWriter}

object Main:
//  @main
  def main(args: Array[String]): Unit =
    println("Hello world!")
//    val fileContents = InputFileLoader.loadFromResources("The_Verdict.txt")
//    println(s"Size of the input file ${fileContents.length}")
//    val splittableRegex = """[,.:;?_~!"()'\s\n\t--]"""
//    val fileContentArray = fileContents
//      .split(s"(?=$splittableRegex)|(?<=$splittableRegex)")
//      .filter(_.nonEmpty)
//    val uniqueWords = fileContentArray.sorted.zipWithIndex
//      .toMap
    val tokenizer = new Tokenizer(){

    }
    tokenizer.buildTokenizerFromRaw("/Users/dooby1/storch/gpt4_vocab_list/o200k_base_vocab_list.txt")
//    tokenizer.buildTokenizer("/Users/dooby1/storch/FulcrumLLM/src/main/resources/The_Verdict.txt")
//    val pw = new PrintWriter(new FileOutputStream("/Users/dooby1/storch/FulcrumLLM/src/main/resources/tokens.json"))
//    val serializedTokens = tokenizer.tokenMap.map {
//        case (t, i) => ("token" -> t.stringify) ~ ("id" -> i)
//    }.map(p => JsonMethods.compact(render(p)))
//    pw.write(serializedTokens.mkString("\n"))
//    pw.close()
    val stringBeforeEncoding = "a wild night. With other punct --{Punct}"
    val enc = tokenizer.encode(stringBeforeEncoding)
    val stringAfterEncoding = tokenizer.decode(enc)
    println(s"Before Encoding: \n $stringBeforeEncoding")
    println(s"Encoding BytePair List:$enc")
    println(s"detokenized and decoded string: $stringAfterEncoding")


