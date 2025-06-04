package com.ssdeep.fulcrum.core

import org.apache.logging.log4j.{LogManager, Logger}


trait ClassNameLogger:
  lazy val logger: Logger = LogManager.getLogger(getClass)
