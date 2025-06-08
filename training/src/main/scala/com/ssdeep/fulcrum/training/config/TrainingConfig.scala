package com.ssdeep.fulcrum.training.config

import com.typesafe.config.Config

case class TrainingConfig(
                           file: String,
                           ratio: Double,
                           limit: Option[Int]
                         )

object TrainingConfig:
  def loadFromConfig(config: Config): TrainingConfig =
    val trainingConfig = config.getConfig("training-config")
    val limit = if trainingConfig.hasPath("limit") then Some(trainingConfig.getInt("limit")) else None
    TrainingConfig(trainingConfig.getString("file"), trainingConfig.getDouble("ratio"),
      limit
      )