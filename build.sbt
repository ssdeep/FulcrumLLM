ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "3.3.5"

resolvers ++= Resolver.sonatypeOssRepos("snapshots")
val json4sNative = "org.json4s" %% "json4s-native" % "4.1.0-M8"

libraryDependencies += json4sNative
libraryDependencies += "me.tongfei" % "progressbar" % "0.10.1"
libraryDependencies ++=  Seq(
  "dev.storch" %% "core" % "0.0-2dfa388-SNAPSHOT",
  "org.bytedeco" % "pytorch-platform" % "2.1.2-1.5.10",
  "org.bytedeco" % "pytorch" % "2.1.2-1.5.10",
  "org.bytedeco" % "pytorch" % "2.1.2-1.5.10" classifier "macosx_arm64", //TODO: make platform agnostic
  "org.bytedeco" % "openblas" % "0.3.26-1.5.10" classifier "macosx_arm64" //TODO: make platform agnostic
)

fork := true

lazy val root = (project in file("."))
  .settings(
    name := "FulcrumLLM"//,
//    idePackagePrefix := Some("com.ssdeep.fulcrum")
  )
