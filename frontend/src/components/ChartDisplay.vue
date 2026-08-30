<template>
  <div class="chart-container">
    <div v-if="loading" class="flex items-center justify-center h-64">
      <span class="text-gray-500">加载图表中...</span>
    </div>
    <div v-else-if="error" class="flex items-center justify-center h-64 text-red-500">
      {{ error }}
    </div>
    <v-chart
      v-else
      :option="chartOption"
      :autoresize="true"
      class="chart"
    />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart, LineChart, PieChart, ScatterChart } from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
} from 'echarts/components'
import type { EChartsOption } from 'echarts'

// Register ECharts components
use([
  CanvasRenderer,
  BarChart,
  LineChart,
  PieChart,
  ScatterChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
])

interface Props {
  chartType?: 'line' | 'bar' | 'pie' | 'scatter'
  title?: string
  data?: {
    labels: string[]
    datasets: {
      name?: string
      data: number[]
      color?: string
    }[]
  }
  loading?: boolean
  error?: string
}

const props = withDefaults(defineProps<Props>(), {
  chartType: 'bar',
  loading: false,
})

const chartOption = computed<EChartsOption>(() => {
  if (!props.data) return {}

  const baseOption: EChartsOption = {
    title: {
      text: props.title || '',
      left: 'center',
    },
    tooltip: {
      trigger: props.chartType === 'pie' ? 'item' : 'axis',
    },
    legend: {
      bottom: 10,
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: props.chartType === 'pie' ? '15%' : '3%',
      containLabel: true,
    },
  }

  if (props.chartType === 'pie') {
    return {
      ...baseOption,
      series: [{
        type: 'pie',
        radius: '50%',
        data: props.data.labels.map((label, index) => ({
          name: label,
          value: props.data!.datasets[0].data[index],
        })),
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowOffsetX: 0,
            shadowColor: 'rgba(0, 0, 0, 0.5)',
          },
        },
      }],
    }
  }

  return {
    ...baseOption,
    xAxis: {
      type: 'category',
      data: props.data.labels,
    },
    yAxis: {
      type: 'value',
    },
    series: props.data.datasets.map((dataset, index) => ({
      name: dataset.name || `数据 ${index + 1}`,
      type: props.chartType,
      data: dataset.data,
      smooth: props.chartType === 'line',
      itemStyle: dataset.color ? { color: dataset.color } : undefined,
    })),
  }
})
</script>

<style scoped>
.chart-container {
  width: 100%;
  height: 400px;
}

.chart {
  width: 100%;
  height: 100%;
}
</style>