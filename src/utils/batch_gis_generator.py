import time
from datetime import datetime, timedelta
from typing import List, Dict

from src.utils.gen_gis_data import GeospatialDataGenerator
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def process_region(
    region: str, data_types: List[str] = ["points", "routes", "polygons"]
) -> None:
    """Process a single region for all specified data types"""
    try:
        logger.info(f"Processing region: {region}")
        for data_type in data_types:
            num_points = 1000 if data_type == "points" else 100
            generator = GeospatialDataGenerator(
                data_type=data_type, region=region, num_points=num_points
            )
            generator.generate_data()
            generator.plot_data()
            generator.save_data(
                format="geojson" if data_type != "polygons" else "shapefile"
            )
            logger.info(f"Completed generating {data_type} for {region}")
    except Exception as e:
        logger.error(f"Error processing {region}: {str(e)}")


def batch_process_regions(batch_size: int = 10, wait_minutes: int = 30) -> None:
    """
    Process regions in batches with rate limiting

    Args:
        batch_size: Number of regions to process in each batch
        wait_minutes: Number of minutes to wait between batches
    """
    # Get all available regions
    all_regions = list(GeospatialDataGenerator._region_queries.keys())
    total_regions = len(all_regions)

    logger.info(f"Starting batch processing of {total_regions} regions")
    logger.info(f"Processing {batch_size} regions every {wait_minutes} minutes")

    # Process regions in batches
    for i in range(0, total_regions, batch_size):
        batch = all_regions[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_regions + batch_size - 1) // batch_size

        logger.info(f"\nStarting batch {batch_num}/{total_batches}")
        logger.info(f"Regions in this batch: {', '.join(batch)}")

        # Process each region in the current batch
        for region in batch:
            process_region(region)

        regions_left = total_regions - (i + len(batch))
        if regions_left > 0:
            next_time = (datetime.now() + timedelta(minutes=wait_minutes)).strftime("%H:%M:%S")
            logger.info(
                f"\nBatch {batch_num} completed. {regions_left} regions remaining."
            )
            logger.info(f"Waiting {wait_minutes} minutes before next batch...")
            logger.info(f"Next batch will start at approximately {next_time}")
            time.sleep(wait_minutes * 60)
        else:
            logger.info("\nAll regions have been processed!")


if __name__ == "__main__":
    # You can adjust these parameters as needed
    BATCH_SIZE = 10  # Number of regions to process in each batch
    WAIT_MINUTES = 30  # Minutes to wait between batches

    batch_process_regions(BATCH_SIZE, WAIT_MINUTES)
