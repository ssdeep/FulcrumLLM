package com.ssdeep.fulcrum.tokenizer

import me.tongfei.progressbar.{ProgressBar, ProgressBarBuilder}

import scala.collection.mutable
import scala.io.{Codec, Source}
import scala.jdk.CollectionConverters.*

case class TokenList(var token: Token):
    var isRoot: Boolean = false
    var next: TokenList = _

case class TokenTrie(token: Token, isRoot: Boolean = false):
    val children: mutable.Map[Token, TokenTrie] = scala.collection.mutable.Map[Token, TokenTrie]()
    var tokenIdMaybe: Option[Long] = None

    def consume(c: Byte): Option[TokenTrie] =
        children.get(UnitToken(c))

    def buildTokenTrie(remainingToken: Token, id: Long): Unit =
      remainingToken match
        case bp: BytePair => children
          .getOrElseUpdate(bp.head, TokenTrie(bp.head))
          .buildTokenTrie(bp.tail, id)
        case t @ UnitToken(c) => children.getOrElseUpdate(t, TokenTrie(t)).tokenIdMaybe = Some(id)
        case _ =>

    def buildTokenTrie(xs: Array[Byte], id: Long): Unit =
      xs match
        case c if c == null || c.isEmpty =>
        case _ =>
          val unitToken = UnitToken(xs.head)
          val trieNode =  children.getOrElseUpdate(unitToken, TokenTrie(unitToken))
            if (xs.nonEmpty && xs.length == 1) {
              trieNode.tokenIdMaybe = Some(id)
            } else if (xs.length > 1) {
              trieNode.buildTokenTrie(xs.tail, id)
            }

class AtomicCounter(var affectedNodes: List[TokenList] = List.empty):
  def add(node: TokenList): Unit =
    affectedNodes = affectedNodes :+ node
object Tokenizer:
  def base64Decode(rawStringFromFile: String): String =
    val bytes = java.util.Base64.getDecoder.decode(rawStringFromFile)
    new String(bytes, "UTF-8")

  def cycleFromRoot(rootNode: TokenList, tokenMap: mutable.Map[Long, Token],  limit: Int = 2000):  Unit = {
    // loop until there is only one node or the vocab limit is reached
    var counter = 0
    
    while (rootNode.next != rootNode && tokenMap.size < limit) {
      val frequency = scala.collection.mutable.Map[Token, AtomicCounter]() // change to heap
      var currentNode = rootNode

      while (!currentNode.next.isRoot) {
        frequency.getOrElseUpdate(BytePair(currentNode.token, currentNode.next.token), AtomicCounter()).add(currentNode)
        currentNode = currentNode.next
      }
      val sortedByFrequency = frequency.toList.sortBy(_._2.affectedNodes.size).reverse
      sortedByFrequency.head match
        case (newToken, nodeList) =>
          tokenMap.getOrElseUpdate(counter, newToken)
          counter = counter + 1
          nodeList.affectedNodes.foreach {
            case node if BytePair(node.token, node.next.token) == newToken =>
              node.token = newToken
              node.next.token = null // invalidate the next token so it cannot be reused
              node.next = node.next.next // collapse next node
            case _ =>
          }
    }
  }

  def buildDefaultO200kBase: Tokenizer =
    buildTokenizerFromRaw("o200k_base.tiktoken")

  def buildTokenizerFromRaw(file: String): Tokenizer =
    val reader = Source.fromResource(file)
      .bufferedReader()
    val lineList = reader
      .lines().toList.asScala
    reader.close()
    val pbBuilder = new ProgressBarBuilder
    pbBuilder.setTaskName("Loading tokenizer file")
    pbBuilder.setInitialMax(lineList.size)
    val tokenTrieRoot = new TokenTrie(UnitToken('*'.toByte), isRoot = true)
    val tokenMap = scala.collection.mutable.Map[Long, Token]()
    ProgressBar.wrap(lineList.zipWithIndex.asJava.iterator, pbBuilder)
      .asScala
      .foreach {
        case (line, index) =>
          val formattedLine = java.util.Base64.getDecoder.decode(line.split("\\s").head) // <vocab_token> <token_id>
          tokenTrieRoot.buildTokenTrie(formattedLine, index)
          val tokenized = Token.makeToken(formattedLine)
          tokenMap.put(index, tokenized)
      }
    Tokenizer(tokenTrieRoot, tokenMap.toMap)

  def buildTokenizer(file: String): Tokenizer =
    val tokenMap = scala.collection.mutable.Map.empty[Long, Token]
    val tokenTrieRoot = new TokenTrie(UnitToken('*'.toByte), isRoot = true)
    val reader = Source.fromFile(file).bufferedReader()
    val lineList = reader.lines().toList.asScala
    val line = lineList.mkString("\n").getBytes("UTF-8")
    reader.close()
    val headToken = UnitToken(line.head)
    val rootToken = TokenList(headToken)
    rootToken.isRoot = true
    val pbBuilder = new ProgressBarBuilder
    pbBuilder.setTaskName("Loading with circular queue")
    pbBuilder.setInitialMax(line.tail.length)
    val lastToken = ProgressBar.wrap(line.tail.iterator.asJava, pbBuilder).asScala
        .foldLeft(rootToken) {
          case (rt, newchar) =>
            rt.next = TokenList(UnitToken(newchar))
            rt.next
        }
    lastToken.next = rootToken // circular queue
    cycleFromRoot(rootToken, tokenMap)
    tokenMap.foreach {
        case (i, tok) =>
          tokenTrieRoot.buildTokenTrie(tok.stringify.getBytes("UTF-8"), i)
    }
    Tokenizer(tokenTrieRoot, tokenMap.toMap)

case class Tokenizer(tokenTrieRoot: TokenTrie, tokenIdMap: Map[Long, Token]):

   private val inverseTokenMap: Map[Token, Long] = tokenIdMap.map((k, v) => (v, k)) // inverse map
   
   def encode(text: String): List[Long] = {
     var currentNode = tokenTrieRoot
     var encodedSoFar = List.empty[Long]
     var (candidateEncoded: Option[Long], consumedSoFar: Array[Byte]) = (None, Array.empty[Byte])
     var charIterator = text.getBytes("UTF-8").iterator
     while(charIterator.hasNext) {
        val c = charIterator.next()
        consumedSoFar = (consumedSoFar.toList :+ c).toArray
        val newNode =  currentNode.consume(c)
        currentNode = newNode match {
          case Some(node) if node.children.isEmpty && node.tokenIdMaybe.nonEmpty =>
            encodedSoFar = encodedSoFar ++ node.tokenIdMaybe
            consumedSoFar = Array()
            candidateEncoded = None
            tokenTrieRoot
          case Some(node) if node.children.nonEmpty && node.tokenIdMaybe.nonEmpty =>
            candidateEncoded = node.tokenIdMaybe
            consumedSoFar = Array()
            node
          case Some(node) if node.children.nonEmpty =>
            node
          case None if candidateEncoded.nonEmpty =>
            encodedSoFar = encodedSoFar ++ candidateEncoded
            val charIteratorList = (consumedSoFar.iterator ++ charIterator).toList
            charIterator = charIteratorList.iterator
            consumedSoFar = Array()
            candidateEncoded = None
            tokenTrieRoot
          case None if currentNode.isRoot =>
            encodedSoFar = encodedSoFar :+ c.toInt
            consumedSoFar = Array()
            candidateEncoded = None
            tokenTrieRoot
          case None =>
            encodedSoFar = encodedSoFar ++ consumedSoFar.flatMap(p => inverseTokenMap.get(UnitToken(p)))
            consumedSoFar = Array()
            candidateEncoded = None
            tokenTrieRoot
          case Some(_) =>
            //
            tokenTrieRoot
        }

     }

     encodedSoFar ++ candidateEncoded ++ consumedSoFar.map {
       c =>
         if (!inverseTokenMap.contains(UnitToken(c))) {
           println(s"Missing: $c")
         }
         inverseTokenMap(UnitToken(c))
     }
   }

   def decode(tokens: List[Long]): String =
     val allBytes = tokens.flatMap {
       c => tokenIdMap(c).bytefy
     }
     new String(allBytes.toArray, "UTF-8")




