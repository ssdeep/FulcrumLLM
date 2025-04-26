package com.ssdeep.fulcrum.tokenizer

import scala.collection.mutable
import scala.io.{Codec, Source}
import scala.jdk.CollectionConverters.*

case class TokenList(var token: Token):
    var isRoot: Boolean = false
    var next: TokenList = _
case class TokenTrie(token: Token, isRoot: Boolean = false):
    val children: mutable.Map[Token, TokenTrie] = scala.collection.mutable.Map[Token, TokenTrie]()
    var tokenIdMaybe: Option[Int] = None
    def consume(c: Char) =
        children.get(UnitToken(c))

    def buildTokenTrie(remainingToken: Token, id: Int): Unit =
      remainingToken match
        case bp: BytePair => children
          .getOrElseUpdate(bp.head, TokenTrie(bp.head))
          .buildTokenTrie(bp.tail, id)
        case t @ UnitToken(c) => children.getOrElseUpdate(t, TokenTrie(t)).tokenIdMaybe = Some(id)
        case _ =>
    def buildTokenTrie(xs: String, id: Int): Unit =
      xs match
        case "" =>
        case _ =>
          val unitToken = UnitToken(xs.head)
          val trieNode =  children.getOrElseUpdate(unitToken, TokenTrie(unitToken))
            trieNode.buildTokenTrie(xs.tail, id)
            if (xs.nonEmpty && xs.length == 1) {
              trieNode.tokenIdMaybe = Some(id)
            } else if (xs.length > 1) {
              trieNode.buildTokenTrie(xs.tail, id)
            }

trait Tokenizer:
   private val limit: Int = 200
   private val toTokenMap: scala.collection.mutable.Map[Int, Token] = scala.collection.mutable.Map.empty
   private var counter: Int = 256
   private val tokenTrieRoot = TokenTrie(UnitToken('*'), true)
   private[tokenizer] val tokenMap: scala.collection.mutable.Map[Token, Int] = scala.collection.mutable.Map.empty
   extension(i: Int) def toChar: Char =  i.asInstanceOf[Char]
   class AtomicCounter(var affectedNodes: List[TokenList] = List.empty):
     def add(node: TokenList): Unit =
       affectedNodes = affectedNodes :+ node

   def cycleFromRoot(rootNode: TokenList): Unit = {
     while(rootNode.next != rootNode && tokenMap.size < limit) {
       val frequency = scala.collection.mutable.Map[Token, AtomicCounter]() // change to heap
       var currentNode = rootNode

       while (!currentNode.next.isRoot) {
         frequency.getOrElseUpdate(BytePair(currentNode.token, currentNode.next.token), AtomicCounter()).add(currentNode)
         currentNode = currentNode.next
       }
       val sortedByFrequency = frequency.toList.sortBy(_._2.affectedNodes.size).reverse
       sortedByFrequency.head match
         case (newToken, nodeList) =>
           tokenMap.getOrElseUpdate(newToken, counter)
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
   def buildTokenizer(file: String): Unit =
     val reader = Source.fromFile(file)
       .bufferedReader()
     val lineList = reader
       .lines().toList.asScala
       .take(1000)
     val line = lineList.mkString("\n")
     reader.close()

       val headToken = UnitToken(line.head)
       val rootToken = TokenList(headToken)
       rootToken.isRoot = true
       val lastToken = line.tail.foldLeft(rootToken) {
             case (rt, newchar) =>
                rt.next = TokenList(UnitToken(newchar))
                rt.next
           }
       lastToken.next = rootToken // circular queue
       cycleFromRoot(rootToken)
       tokenMap.foreach {
         case (tok, i) => toTokenMap.put(i, tok)
           tokenTrieRoot.buildTokenTrie(tok.stringify, i)
       }

   def encode(text: String): List[Int] = {
     var currentNode = tokenTrieRoot
     var encodedSoFar = List.empty[Int]
     var (candidateEncoded: Option[Int], consumedSoFar: String) = (None, "")
     var charIterator = text.toCharArray.iterator
     while(charIterator.hasNext) {
        val c = charIterator.next()
        consumedSoFar = consumedSoFar + c
        val newNode =  currentNode.consume(c)
        currentNode = newNode match {
          case Some(node) if node.children.isEmpty && node.tokenIdMaybe.nonEmpty =>
            encodedSoFar = encodedSoFar ++ node.tokenIdMaybe
            consumedSoFar = ""
            candidateEncoded = None
            tokenTrieRoot
          case Some(node) if node.children.nonEmpty && node.tokenIdMaybe.nonEmpty =>
            candidateEncoded = node.tokenIdMaybe
            consumedSoFar = ""
            node
          case Some(node) if node.children.nonEmpty =>
            node
          case None if candidateEncoded.nonEmpty =>
            encodedSoFar = encodedSoFar ++ candidateEncoded
            val charIteratorList = (consumedSoFar.toCharArray.iterator ++ charIterator).toList
            charIterator = charIteratorList.iterator
            consumedSoFar = ""
            candidateEncoded = None
            tokenTrieRoot
          case None if currentNode.isRoot =>
            encodedSoFar = encodedSoFar :+ c.toInt
            consumedSoFar = ""
            candidateEncoded = None
            tokenTrieRoot
          case None =>
            encodedSoFar = encodedSoFar ++ consumedSoFar.toCharArray.map(_.toInt)
            consumedSoFar = ""
            candidateEncoded = None
            tokenTrieRoot
        }

     }
     encodedSoFar
   }

   def decode(tokens: List[Int]): String = tokens.map {
       case c if c <= 255 => UnitToken(c.toChar)
       case c => toTokenMap(c)
    }
     .map(_.stringify).mkString("")




