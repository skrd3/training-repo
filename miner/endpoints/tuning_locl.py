import os
from datetime import datetime
from datetime import timedelta

import toml
import yaml
import json
import redis
import httpx
import requests
from fastapi import Depends
from fastapi import HTTPException
from fastapi.routing import APIRouter
from fiber.logging_utils import get_logger
from fiber.miner.core.configuration import Config
from fiber.miner.dependencies import blacklist_low_stake
from fiber.miner.dependencies import get_config
from fiber.miner.dependencies import verify_get_request
from fiber.miner.dependencies import verify_request
from pydantic import ValidationError

import core.constants as cst
from core.models.payload_models import MinerTaskOffer
from core.models.payload_models import MinerTaskResponse
from core.models.utility_models import MinerSubmission
from validator.utils.hash_verification import calculate_model_hash
from core.models.payload_models import TrainingRepoResponse
from core.models.payload_models import TrainRequestGrpo
from core.models.payload_models import TrainRequestImage
from core.models.payload_models import TrainRequestText
from core.models.payload_models import TrainResponse
from core.models.utility_models import FileFormat
from core.models.utility_models import TaskType
from core.models.tournament_models import TournamentType
from core.utils import download_s3_file
from miner.config import WorkerConfig
from miner.dependencies import get_worker_config
from miner.logic.job_handler import create_job_diffusion
from miner.logic.job_handler import create_job_text
from core.models.utility_models import ImageModelType


logger = get_logger(__name__)

# Redis配置 - 建议移到配置文件中
REDIS_CONFIG = {
    'host': '77.90.43.51',
    'port': 36380,
    'db': 6,
    'password': 'ddXpooDFEW#SD',
    'decode_responses': True
}

# 线程安全的作业完成时间管理
import threading
_job_finish_time_lock = threading.Lock()
_current_job_finish_time = None


def get_current_job_finish_time():
    """线程安全地获取当前作业完成时间"""
    with _job_finish_time_lock:
        return _current_job_finish_time


def set_current_job_finish_time(finish_time):
    """线程安全地设置当前作业完成时间"""
    global _current_job_finish_time
    with _job_finish_time_lock:
        _current_job_finish_time = finish_time


async def save_training_params(task_id: str, train_request):
    """保存训练参数"""
    try:
        params_dir = os.path.join(cst.CONFIG_DIR, "training_params")
        os.makedirs(params_dir, exist_ok=True)
        file_path = os.path.join(params_dir, f"{task_id}.json")
        with open(file_path, "w") as f:
            json.dump(train_request.model_dump(), f)
        logger.info(f"Training parameters saved for task {task_id}")
    except Exception as e:
        logger.error(f"Error saving training parameters for task {task_id}: {str(e)}")
        raise


async def tune_model_text(
    train_request: TrainRequestText,
    worker_config: WorkerConfig = Depends(get_worker_config),
):
    logger.info("Starting model tuning.")

    set_current_job_finish_time(datetime.now() + timedelta(hours=train_request.hours_to_complete))
    logger.info(f"Job received is {train_request}")

    try:
        logger.info(train_request.file_format)
        if train_request.file_format != FileFormat.HF:
            if train_request.file_format == FileFormat.S3:
                train_request.dataset = await download_s3_file(train_request.dataset)
                logger.info(train_request.dataset)
                train_request.file_format = FileFormat.JSON

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job = create_job_text(
        job_id=str(train_request.task_id),
        dataset=train_request.dataset,
        model=train_request.model,
        dataset_type=train_request.dataset_type,
        file_format=train_request.file_format,
        expected_repo_name=train_request.expected_repo_name,
    )
    logger.info(f"Created job {job}")
    worker_config.trainer.enqueue_job(job)

    return {"message": "Training job enqueued.", "task_id": job.job_id}


async def tune_model_grpo(
    train_request: TrainRequestGrpo,
    worker_config: WorkerConfig = Depends(get_worker_config),
):
    logger.info("Starting model tuning.")

    set_current_job_finish_time(datetime.now() + timedelta(hours=train_request.hours_to_complete))
    logger.info(f"Job received is {train_request}")

    try:
        logger.info(train_request.file_format)
        if train_request.file_format != FileFormat.HF:
            if train_request.file_format == FileFormat.S3:
                train_request.dataset = await download_s3_file(train_request.dataset)
                logger.info(train_request.dataset)
                train_request.file_format = FileFormat.JSON

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job = create_job_text(
        job_id=str(train_request.task_id),
        dataset=train_request.dataset,
        model=train_request.model,
        dataset_type=train_request.dataset_type,
        file_format=train_request.file_format,
        expected_repo_name=train_request.expected_repo_name,
    )
    logger.info(f"Created job {job}")
    worker_config.trainer.enqueue_job(job)

    return {"message": "Training job enqueued.", "task_id": job.job_id}


async def tune_model_diffusion(
    train_request: TrainRequestImage,
    worker_config: WorkerConfig = Depends(get_worker_config),
):
    logger.info("Starting model tuning.")

    set_current_job_finish_time(datetime.now() + timedelta(hours=train_request.hours_to_complete))
    logger.info(f"Job received is {train_request}")
    try:
        train_request.dataset_zip = await download_s3_file(
            train_request.dataset_zip, f"{cst.DIFFUSION_DATASET_DIR}/{train_request.task_id}.zip"
        )
        logger.info(train_request.dataset_zip)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job = create_job_diffusion(
        job_id=str(train_request.task_id),
        dataset_zip=train_request.dataset_zip,
        model=train_request.model,
        model_type=train_request.model_type,
        expected_repo_name=train_request.expected_repo_name,
    )
    logger.info(f"Created job {job}")
    worker_config.trainer.enqueue_job(job)

    return {"message": "Training job enqueued.", "task_id": job.job_id}



async def retune_model_diffusion(
    task_id: str,
    worker_config: WorkerConfig = Depends(get_worker_config),
):
    """根据task_id重新调用训练"""
    if not task_id or not isinstance(task_id, str):
        raise HTTPException(status_code=400, detail="Invalid task_id format")
        
    params_dir = os.path.join(cst.CONFIG_DIR, "training_params")
    file_path = os.path.join(params_dir, f"{task_id}.json")
    
    try:
        with open(file_path, "r") as f:
            try:
                params = json.load(f)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON format in parameters file: {str(e)}")
            
            try:
                train_request = TrainRequestImage(**params)
            except ValidationError as e:
                raise HTTPException(status_code=400, detail=f"Invalid parameters format: {str(e)}")               
            # 调用外部的/dc_start_training_image/接口
            worker_list_key = '44_sdxl_img_worker'
            if train_request.model_type != ImageModelType.SDXL:
                worker_list_key = 'flux_img_worker'

            # 对于重新训练，我们需要确保train_request有正确的task_id
            if not hasattr(train_request, 'task_id') or not train_request.task_id:
                train_request.task_id = task_id
            
            workerurl = getWorkerUrl(train_request, worker_list_key)
            logger.info(f"workerurl==== {workerurl}")
            
            if not workerurl:
                raise HTTPException(status_code=503, detail="No available worker URL found")
                
            req_job_url = f"http://{workerurl}/dc_start_training_image/"
            
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    logger.info(f"Sending POST request to: {req_job_url}")
                    response = await client.post(
                        req_job_url,
                        json=train_request.model_dump(),
                        headers={"Content-Type": "application/json"}
                    )
                    logger.info(f"Response status: {response.status_code}")
                    response.raise_for_status()
                    return response.json()
            except httpx.TimeoutException as e:
                logger.error(f"HTTP request timeout: {str(e)}")
                raise HTTPException(status_code=504, detail=f"Request timeout to worker: {str(e)}")
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP status error: {e.response.status_code} - {e.response.text}")
                raise HTTPException(status_code=502, detail=f"Worker returned error: {e.response.status_code}")
            except httpx.RequestError as e:
                logger.error(f"HTTP request error: {str(e)}")
                raise HTTPException(status_code=503, detail=f"Failed to connect to worker: {str(e)}")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No training parameters found for task {task_id}")
    except Exception as e:
        logger.error(f"Error retuning model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retuning model: {str(e)}")



async def mut_tune_model_diffusion(
    train_request: TrainRequestImage,
    worker_config: WorkerConfig = Depends(get_worker_config),
):
    logger.info(f"Starting model tuning {train_request}")
    try:
        # 保存训练参数
        await save_training_params(str(train_request.task_id), train_request)
    except Exception as e:
        logger.error(f"Error saving training parameters: {str(e)}")
        # 继续执行训练流程，不中断

    set_current_job_finish_time(datetime.now() + timedelta(hours=train_request.hours_to_complete))
    logger.info(f"Job received is {train_request}")
    
    # 调用外部的/dc_start_training_image/接口
    worker_list_key = '44_sdxl_img_worker'
    if train_request.model_type != ImageModelType.SDXL:
        worker_list_key = 'flux_img_worker'
    
    try:
        workerurl = getWorkerUrl(train_request, worker_list_key)
        logger.info(f"workerurl==== {workerurl}")
        
        if not workerurl:
            raise HTTPException(status_code=503, detail="No available workers found")
            
        req_job_url = f"http://{workerurl}/dc_start_training_image/"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            logger.info(f"Sending POST request to: {req_job_url}")
            response = await client.post(
                req_job_url,
                json=train_request.model_dump(),
                headers={"Content-Type": "application/json"}
            )
            logger.info(f"Response status: {response.status_code}")
            response.raise_for_status()
            return response.json()
    except httpx.TimeoutException as e:
        logger.error(f"HTTP request timeout: {str(e)}")
        raise HTTPException(status_code=504, detail=f"Request timeout to worker: {str(e)}")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP status error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=502, detail=f"Worker returned error: {e.response.status_code}")
    except httpx.RequestError as e:
        logger.error(f"HTTP request error: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Failed to connect to worker: {str(e)}")
    except Exception as e:
        logger.error(f"Error retuning model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retuning model: {str(e)}")


async def get_latest_model_submission(task_id: str) -> MinerSubmission:
    logger.info(f"get_latest_model_submission req task_id==== {task_id}")
    params_dir = os.path.join(cst.CONFIG_DIR, "training_params")
    req_file_path = os.path.join(params_dir, f"{task_id}.json")
    
    try:
        with open(req_file_path, "r") as f:
            try:
                params = json.load(f)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON format in parameters file: {str(e)}")
            
            train_request = TrainRequestImage(**params)
            repo_res_value = f"skrd3/{train_request.expected_repo_name}"
            
        if repo_res_value:
            logger.info(f"get repo_res_value from local req==== {repo_res_value}")
            model_hash = calculate_model_hash(repo_res_value)
            return MinerSubmission(repo=repo_res_value, model_hash=model_hash)
        else:
            logger.info(f"can't get repo_res_value from local req==== {task_id}")
            raise HTTPException(status_code=404, detail=f"No model submission found for task {task_id}")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No training parameters found for task {task_id}")
    except Exception as e:
        logger.error(f"Error getting latest model submission: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting latest model submission: {str(e)}")


async def get_latest_old_model_submission(task_id: str) -> MinerSubmission:
    try:
        config_filename = f"{task_id}.yml"
        config_path = os.path.join(cst.CONFIG_DIR, config_filename)
        repo_id = None
        
        if os.path.exists(config_path):
            with open(config_path, "r") as file:
                config_data = yaml.safe_load(file)
                repo_id = config_data.get("hub_model_id", None)
        else:
            config_filename = f"{task_id}.toml"
            config_path = os.path.join(cst.CONFIG_DIR, config_filename)
            with open(config_path, "r") as file:
                config_data = toml.load(file)
                repo_id = config_data.get("huggingface_repo_id", None)

        if repo_id is None:
            raise HTTPException(status_code=404, detail=f"No model submission found for task {task_id}")

        model_hash = calculate_model_hash(repo_id)
        
        return MinerSubmission(repo=repo_id, model_hash=model_hash)

    except FileNotFoundError as e:
        logger.error(f"No submission found for task {task_id}: {str(e)}")
        raise HTTPException(status_code=404, detail=f"No model submission found for task {task_id}")
    except Exception as e:
        logger.error(f"Error retrieving latest model submission for task {task_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving latest model submission: {str(e)}",
        )


def getWorkerUrl(train_request, worker_list_key: str):
    """
    获取可用工作器的URL
    查询Redis中的工作器列表，找到空闲的工作器并返回其URL
    """
    try:
        # 连接到Redis
        r = redis.Redis(**REDIS_CONFIG)
        
        # 使用传入的worker_list_key
        worker_list = r.lrange(worker_list_key, 0, -1)
        
        logger.info(f"Found {len(worker_list)} workers in {worker_list_key} list: {worker_list}")
        
        if not worker_list:
            logger.info(f"No workers found in {worker_list_key} list")
            return ""
        
        # 遍历工作器列表，检查每个工作器的状态
        for worker_key in worker_list:
            worker_data = r.get(worker_key)
            if worker_data is None:
                # 找到没有数据的工作器，说明该工作器空闲
                logger.info(f"Found available worker: {worker_key}")
                
                # 拼接URL字符串作为key查询对应的值
                url_key = f"{worker_key}_url"
                worker_url = r.get(url_key)
                
                if worker_url:
                    req_url = f"http://{worker_url}/"
                    # 检查worker URL是否可以访问，返回404状态码
                    try:
                        response = requests.get(req_url, timeout=5)
                        if response.status_code == 404:
                            logger.info(f"Found available worker: {worker_key}, URL accessible: {worker_url}")
                            
                            exptime = train_request.hours_to_complete * 3600
                            task_id = train_request.task_id
                            repo_res_key = f"42:res:{task_id}"
                            repo_res_value = f"skrd3/{train_request.expected_repo_name}"
                            
                            # 设置worker_key为task_id，过期时间为exptime秒
                            r.set(worker_key, task_id, ex=int(exptime))
                            r.set(f"44:{task_id}", worker_url, ex=int(exptime * 3))
                            r.set("44:task_exits", "1", ex=int(exptime * 24))
                            r.set(repo_res_key, repo_res_value, ex=int(exptime * 24))

                            return worker_url
                    except requests.exceptions.RequestException as e:
                        logger.warning(f"Worker {worker_key} URL {worker_url} not accessible: {str(e)}")
                        continue
                else:
                    logger.warning(f"No URL found for worker: {worker_key} with key: {url_key}")
                    continue
            else:
                logger.info(f"Worker {worker_key} is busy: {worker_data}")
        
        # 没有找到可用的工作器或URL
        logger.info("No available worker URL found")
        return ""
        
    except redis.ConnectionError as e:
        logger.error(f"Redis connection error: {str(e)}")
        return ""
    except Exception as e:
        logger.error(f"Error getting worker URL: {str(e)}")
        return ""


def checkAccept(worker_list_key: str):
    """
    检查是否有可用的工作器
    查询Redis中的工作器列表，找到空闲的工作器则返回True
    """
    try:
        # 连接到Redis
        r = redis.Redis(**REDIS_CONFIG)
        
        worker_list = r.lrange(worker_list_key, 0, -1)
        logger.info(f"Found {len(worker_list)} workers in {worker_list_key} list: {worker_list}")
        
        if not worker_list:
            logger.info(f"No workers found in worker_list_key= {worker_list_key} list")
            return False
        
        # 遍历工作器列表，检查每个工作器的状态
        for worker_key in worker_list:
            worker_data = r.get(worker_key)
            if worker_data is None:
                # 找到没有数据的工作器，说明该工作器空闲
                worker_url = r.get(f"{worker_key}_url")
                
                # 检查worker URL是否存在
                if not worker_url:
                    logger.warning(f"No URL found for worker: {worker_key}")
                    continue
                    
                req_url = f"http://{worker_url}/"

                # 检查worker URL是否可以访问，返回404状态码
                try:
                    response = requests.get(req_url, timeout=5)
                    if response.status_code == 404:
                        logger.info(f"Found available worker: {worker_key}, URL accessible: {req_url}")
                        return True
                    else:
                        logger.warning(f"Worker {worker_key} URL {req_url} returned status {response.status_code}")
                        continue
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Worker {worker_key} URL {req_url} not accessible: {str(e)}")
                    continue
            else:
                logger.info(f"Worker {worker_key} is busy: {worker_data}")
        
        # 所有工作器都有数据（都在忙），返回False
        logger.info("All workers are busy")
        return False
        
    except redis.ConnectionError as e:
        logger.error(f"Redis connection error: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error checking worker availability: {str(e)}")
        return False
    finally:
        # 确保函数总是返回布尔值
        pass



async def task_offer(
    request: MinerTaskOffer,
    config: Config = Depends(get_config),
    worker_config: WorkerConfig = Depends(get_worker_config),
) -> MinerTaskResponse:
    try:
        logger.info(f"An offer has come through {request}")
        current_time = datetime.now()
        current_job_finish_time = get_current_job_finish_time()
        
        if request.task_type not in [TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK, TaskType.GRPOTASK, TaskType.CHATTASK]:
            return MinerTaskResponse(
                message=f"This endpoint only accepts text tasks: "
                f"{TaskType.INSTRUCTTEXTTASK}, {TaskType.DPOTASK}, {TaskType.GRPOTASK} and {TaskType.CHATTASK}",
                accepted=False,
            )

        if "llama" not in request.model.lower():
            return MinerTaskResponse(message="I'm not yet optimised and only accept llama-type jobs", accepted=False)

        worker_list_key = '44_instruct_worker'
        logger.info(f"checkAccept check worker running ")
        if checkAccept(worker_list_key):
            
            if current_job_finish_time is None or current_time + timedelta(hours=1) > current_job_finish_time:
                if request.hours_to_complete < 13 and request.hours_to_complete > 5:
                    logger.info(f"Accepting the offer - type {request} snr")
                    return MinerTaskResponse(message=f"Yes. I can do {request.task_type} jobs", accepted=True)
                else:
                    logger.info("Rejecting offer")
                    return MinerTaskResponse(message="I only accept small jobs", accepted=False)
            else:
                return MinerTaskResponse(
                    message=f"Currently busy with another job until {current_job_finish_time.isoformat()}",
                    accepted=False,
                )
        else:
            return MinerTaskResponse(
                message="Currently busy with another job ",
                accepted=False,
            )
    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in task_offer: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing task offer: {str(e)}")


async def task_offer_image(
    request: MinerTaskOffer,
    config: Config = Depends(get_config),
    worker_config: WorkerConfig = Depends(get_worker_config),
) -> MinerTaskResponse:
    try:
        logger.info(f"An image offer has come through {request}")
        current_time = datetime.now()
        
        try:
            r = redis.Redis(**REDIS_CONFIG)
            hour_limit = 2
            task_exits = r.get("44:task_exits")
            if task_exits is not None:
                hour_limit = 1
        except redis.ConnectionError as e:
            logger.error(f"Redis connection error: {str(e)}")
            hour_limit = 2  # 使用默认值
            
        if request.task_type != TaskType.IMAGETASK:
            return MinerTaskResponse(message="This endpoint only accepts image tasks", accepted=False)
            
        if "flux" in request.model.lower():
            logger.info(f"Rejecting offer - {request.model} is not sdxl")
            return MinerTaskResponse(message="I'm not yet optimised this jobs", accepted=False)
            
        # 排除ifmain/UltraReal_Fine-Tune 也是flux模型
        if "ultrareal" in request.model.lower():
            logger.info(f"Rejecting offer - {request.model} is not sdxl")
            return MinerTaskResponse(message="I'm not yet optimised this jobs", accepted=False)
            
        worker_list_key = '44_sdxl_img_worker'
        logger.info(f"checkAccept check worker running ")
        
        if checkAccept(worker_list_key):
            base_weights_dir = f"{cst.OUTPUT_DIR}/base/"
            model_safe_name = request.model.replace("/", "_")
            weight_file_path = os.path.join(base_weights_dir, f"{model_safe_name}.safetensors")
            
            # 检查权重文件是否存在
            if os.path.exists(weight_file_path):
                logger.info(f"Accepting the image offer {request}")
                if request.hours_to_complete < 3 and request.hours_to_complete >= hour_limit:
                    logger.info(f"Accepting the image offer {request}")
                    return MinerTaskResponse(message="Yes. I can do image jobs", accepted=True)
                else:
                    logger.info("Rejecting offer - too long")
                    return MinerTaskResponse(message="I only accept small jobs", accepted=False)
            else:
                logger.info(f"Weight file not found: {weight_file_path}")
                return MinerTaskResponse(message="Required model weights not available", accepted=False)
        else:
            return MinerTaskResponse(
                message="Currently busy with another job ",
                accepted=False,
            )

    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in task_offer_image: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing task offer: {str(e)}")


async def get_training_repo(task_type: TournamentType) -> TrainingRepoResponse:
    return TrainingRepoResponse(
        github_repo="https://github.com/skrd3/training-repo", 
        commit_hash="bba50114dcea8ece2b0b18d7424eab42de6af0cd"
    )


def factory_router() -> APIRouter:
    router = APIRouter()
    router.add_api_route(
        "/task_offer/",
        task_offer,
        tags=["Subnet"],
        methods=["POST"],
        response_model=MinerTaskResponse,
        dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    )

    router.add_api_route(
        "/task_offer_image/",
        task_offer_image,
        tags=["Subnet"],
        methods=["POST"],
        response_model=MinerTaskResponse,
        dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    )

    router.add_api_route(
        "/get_latest_model_submission/{task_id}",
        get_latest_model_submission,
        tags=["Subnet"],
        methods=["GET"],
        response_model=MinerSubmission,
        summary="Get Latest Model Submission",
        description="Retrieve the latest model submission for a given task ID",
        dependencies=[Depends(blacklist_low_stake), Depends(verify_get_request)],
    )

    router.add_api_route(
        "/training_repo/{task_type}",
        get_training_repo,
        tags=["Subnet"],
        methods=["GET"],
        response_model=TrainingRepoResponse,
        summary="Get Training Repo",
        description="Retrieve the training repository and commit hash for the tournament.",
        dependencies=[Depends(blacklist_low_stake), Depends(verify_get_request)],
    )

    router.add_api_route(
        "/start_training/",  # TODO: change to /start_training_text or similar
        tune_model_text,
        tags=["Subnet"],
        methods=["POST"],
        response_model=TrainResponse,
        dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    )
    router.add_api_route(
        "/start_training_grpo/",
        tune_model_grpo,
        tags=["Subnet"],
        methods=["POST"],
        response_model=TrainResponse,
        dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    )
    router.add_api_route(
        "/start_training_image/",
        mut_tune_model_diffusion,
        tags=["Subnet"],
        methods=["POST"],
        response_model=TrainResponse,
        dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    )
    router.add_api_route(
        "/retune_model_diffusion/{task_id}",
        retune_model_diffusion,
        tags=["Subnet"],
        methods=["POST"],
        response_model=TrainResponse,
        #dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
        description="重新发送已记录的imge训练请求（开发调试专用，无安全校验）",
    )

    return router