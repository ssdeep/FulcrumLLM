package com.ssdeep.fulcrum.tokenizer

object InputFileLoader:
  def loadFromResources(file: String): String =
    import scala.io._
    val fileContents = Source.fromResource(file)
    try {
      fileContents.getLines().mkString("\n")
    } finally {
      fileContents.close()
    }
    
    
