import time
from datetime import datetime, timedelta
from typing import List, Dict

from src.utils.gen_gis_data import GeospatialDataGenerator
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

DATA_CONFIGS = {
    "pois": {"num_points": 1000, "format": "geojson"},
    "routes": {"num_points": 100, "format": "geojson"},
    "polygons": {"num_points": 50, "format": "shapefile"},
}


def process_region(region: str, data_types: List[str] = None) -> None:
    """Process a single region for all specified data types"""
    try:
        logger.info(f"Processing region: {region}")

        # Create a single generator instance to reuse cached data
        generator = GeospatialDataGenerator(region=region)

        # Process each data type
        for data_type in data_types:
            try:
                logger.info(f"\nProcessing {data_type} for {region}...")

                # Update generator configuration
                generator.data_type = data_type
                generator.num_points = DATA_CONFIGS[data_type]["num_points"]

                # Generate, plot, and save data
                generator.generate_data()
                generator.plot_data()
                generator.save_data(format=DATA_CONFIGS[data_type]["format"])

                logger.info(f"Successfully processed {data_type} for {region}")

            except Exception as e:
                logger.error(f"Error processing {data_type} for {region}: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Error processing region {region}: {str(e)}")


def batch_process_regions(
    batch_size: int = 10, wait_minutes: int = 30, data_types: List[str] = None
) -> None:
    """Process regions in batches with rate limiting"""
    # Get all available regions
    all_regions = list(GeospatialDataGenerator._region_queries.keys())
    total_regions = len(all_regions)

    # Validate data types
    if data_types:
        invalid_types = set(data_types) - set(DATA_CONFIGS.keys())
        if invalid_types:
            raise ValueError(f"Invalid data types: {invalid_types}")
    else:
        data_types = list(DATA_CONFIGS.keys())

    logger.info(f"Starting batch processing of {total_regions} regions")
    logger.info(f"Data types to generate: {data_types}")
    logger.info(f"Processing {batch_size} regions every {wait_minutes} minutes")

    start_time = datetime.now()

    # Process regions in batches
    for i in range(0, total_regions, batch_size):
        batch = all_regions[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_regions + batch_size - 1) // batch_size

        logger.info(f"\nStarting batch {batch_num}/{total_batches}")
        logger.info(f"Regions in this batch: {', '.join(batch)}")

        batch_start = datetime.now()

        # Process each region in the current batch
        for region in batch:
            process_region(region, data_types)

        batch_duration = datetime.now() - batch_start
        regions_left = total_regions - (i + len(batch))

        if regions_left > 0:
            next_time = (datetime.now() + timedelta(minutes=wait_minutes)).strftime(
                "%H:%M:%S"
            )
            logger.info(f"\nBatch {batch_num} completed in {batch_duration}")
            logger.info(f"{regions_left} regions remaining")
            logger.info(f"Waiting {wait_minutes} minutes before next batch...")
            logger.info(f"Next batch will start at approximately {next_time}")
            time.sleep(wait_minutes * 60)
        else:
            total_duration = datetime.now() - start_time
            logger.info("\nAll regions have been processed!")
            logger.info(f"Total processing time: {total_duration}")


if __name__ == "__main__":
    BATCH_SIZE = 20
    WAIT_MINUTES = 30
    DATA_TYPES = ["pois", "routes", "polygons"]

    batch_process_regions(
        batch_size=BATCH_SIZE, wait_minutes=WAIT_MINUTES, data_types=DATA_TYPES
    )
