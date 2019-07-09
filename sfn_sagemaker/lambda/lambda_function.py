import boto3
import logging
import json


def lambda_handler(event, context):
    client = boto3.client('sagemaker')
    response = client.create_model(
        ModelName=event["model_name"],
        PrimaryContainer={
            'ContainerHostname': event["container_host"],
            'Image': event["container_image"],
            'ModelDataUrl': event["model_uri"],
            'Environment':event["env_model"]
            },
        ExecutionRoleArn=event["role"]
        )
    return response
    
        