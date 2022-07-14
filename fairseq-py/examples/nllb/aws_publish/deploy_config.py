# Copyright (c) Facebook, Inc. and its affiliates.
import os

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "aws credentials required")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "aws credentials required")
AWS_SESSION_TOKEN = os.environ.get("AWS_SESSION_TOKEN", "aws credentials required")

config = {
    "aws_access_key_id": AWS_ACCESS_KEY_ID,
    "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
    "aws_session_token": AWS_SESSION_TOKEN,
    "aws_region": "us-east-1",
    # @nocommit: secret don't open source !
    "account": "022599147766",
    "arn_role": "arn:aws:iam::022599147766:role/AmazonSageMaker-ExecutionRole-20211022",
    "sagemaker_role": "AmazonSageMaker-ExecutionRole-20211022",
}
