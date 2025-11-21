from dagster import ScheduleDefinition, DefaultScheduleStatus

from .jobs import job_compare_models

schedule_tuesday_saturday_0005 = ScheduleDefinition(
    job=job_compare_models,
    cron_schedule='5 0 * * 2-6', # 00:05 Tue–Sat
    execution_timezone='Europe/Berlin',
    default_status=DefaultScheduleStatus.RUNNING
)
