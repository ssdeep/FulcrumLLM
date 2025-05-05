import Dependencies._
ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "3.3.5"

ThisBuild / resolvers ++= Resolver.sonatypeOssRepos("snapshots")

fork := true

lazy val root = (project in file("."))
  .settings(
    name := "FulcrumLLM"//,
  )
  .aggregate(tokenizer)

lazy val core = (project in file("core"))
  .settings(
    name := "core"
  )

lazy val tokenizer = (project in file("tokenizer"))
  .settings(
    name := "tokenizer",
    libraryDependencies ++= (json4sNative+: log4j),
  )
  .dependsOn(core)
  .aggregate(core)

lazy val attention = (project in file("attention"))
  .settings(
    name := "attention"
  )
  .dependsOn(tokenizer)
  .aggregate(tokenizer)

