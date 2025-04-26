package com.ssdeep.fulcrum.tokenizer

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
    tokenizer.buildTokenizer("/Users/dooby1/Downloads/wikiarchive/AllCombined.txt")
    val stringBeforeEncoding = "a wild night. With other punct --{Punct}"
    val enc = tokenizer.encode(stringBeforeEncoding)
    val stringAfterEncoding = tokenizer.decode(enc)
    println(s"Before Encoding: \n $stringBeforeEncoding")
    println(s"Encoding BytePair List:$enc")
    println(s"detokenized and decoded string: $stringAfterEncoding")


