import Dependencies._
libraryDependencies ++= log4j
libraryDependencies += "me.tongfei" % "progressbar" % "0.10.1"
libraryDependencies ++=  Seq(
  "dev.storch" %% "core" % "0.0-2dfa388-SNAPSHOT",
  //  "org.bytedeco" % "pytorch-platform" % "2.1.2-1.5.10",
  "org.bytedeco" % "pytorch" % "2.1.2-1.5.10" classifier "macosx-arm64", //TODO: make platform agnostic
  "org.bytedeco" % "openblas" % "0.3.26-1.5.10" classifier "macosx-arm64", //TODO: make platform agnostic
//  "org.bytedeco" % "pytorch-platform-gpu" % "2.1.2-1.5.10" classifier "macosx-arm64", //TODO: make platform agnostic
//  "org.bytedeco" % "cuda-platform-redist" % "12.3-8.9-1.5.10" classifier "macosx-arm64",
//  "org.bytedeco" % "cuda" % "12.3-8.9-1.5.10",
//  "org.bytedeco" % "cuda" % "12.3-8.9-1.5.10" classifier "macosx-arm64",
//  "org.bytedeco" % "cuda" % "12.3-8.9-1.5.10" classifier "macosx-arm64-redist"
)