module.exports = {
    themeConfig: {
        logo: './img/leteasyAi.svg',
        //侧边拦
        sidebar: [
            {
                title: 'leteasyAi数据分析',   // 必要的
                path :'/',
                collapsable: false, // 可选的, 默认值是 true,
                sidebarDepth: 1,    // 可选的, 默认值是 1
                children : [
                    'svm'
                ]
            }
        ]
    },
    description: '数据分析',
    base: '/doc/',
    head: [
        [
            'script',
            {
                src: '/assets/js/index.js'
            }
        ]
    ],
    plugins: [
        '@vuepress/back-to-top',
        '@vuepress/nprogress'
    ]
}
