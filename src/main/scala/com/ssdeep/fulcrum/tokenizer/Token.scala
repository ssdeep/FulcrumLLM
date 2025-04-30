package com.ssdeep.fulcrum.tokenizer

/**
 * Token trait is the fundamental unit of LLMs and is broadly divided into three sub-types
 * - BytePair(first, second) => is a token made up of two parts, first and second, each can be a token
 * - UnitToken(c: Byte) => is the smallest individual byte that makes up a token
 * - Empty => is a null case that signifies an empty token. Just here for completeness
 */
sealed trait Token:
  lazy val head: Token = this match
    case BytePair(first, second) => first.head
    case a @ UnitToken(c) => a
    case Empty => Empty
  
  lazy val tail: Token = this match
    case BytePair(first: UnitToken, second) => second
    case BytePair(first: BytePair, second) => BytePair(first.tail, second)
    case UnitToken(c) => Empty
    case Empty => Empty
    case BytePair(Empty, Empty) => Empty
    case BytePair(Empty, second: Token) => second.tail
    
  def isMatch(s: String): Boolean =
    this.stringify == s

case class BytePair(first: Token, second: Token) extends Token
case class UnitToken(c: Byte) extends Token
case object Empty extends Token

object Token:
   extension(t: Token) def stringify: String =
     new String(t.bytefy, "UTF-8")
   extension(t: Token) def bytefy: Array[Byte] =
     t match
       case BytePair(first, second) => first.bytefy ++ second.bytefy
       case UnitToken(c) => Array(c)
       case Empty => Array.empty

   def makeToken(xs: Array[Byte]): Token = {
      xs match {
        case arrBytes if arrBytes.isEmpty => Empty
        case arrBytes if arrBytes.length == 1 => UnitToken(arrBytes.head)
        case arrBytes => BytePair(UnitToken(arrBytes.head), makeToken(arrBytes.tail))
      }
   }
