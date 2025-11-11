from dagster import ScheduleDefinition, DefaultScheduleStatus

from .jobs import job_daily_0800

schedule_0800 = ScheduleDefinition(
    job=job_daily_0800,
    cron_schedule='00 8 * * *',
    execution_timezone='Europe/Berlin',
    default_status=DefaultScheduleStatus.RUNNING
)