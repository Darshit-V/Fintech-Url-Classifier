import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule


class MyCrawler(CrawlSpider):
    name = 'mycrawler'
    allowed_domains = ['indiatoday.in']
    start_urls = ['https://www.indiatoday.in/']  

    rules = (
        Rule(LinkExtractor(), callback='parse_item', follow=True),
    )

    custom_settings = {
        'DEPTH_LIMIT': 2,
        'SCHEDULER_DISK_QUEUE': 'scrapy.squeues.PickleFifoDiskQueue',
        'SCHEDULER_MEMORY_QUEUE': 'scrapy.squeues.FifoMemoryQueue'
    }

    def parse_item(self, response):
        self.logger.info('Parsing URL: %s', response.url)
        # Extracting links and storing them in a file
        with open('links.txt', 'a') as f:
            f.write(response.url + '\n')
        # You can extract data or perform actions here
        pass
