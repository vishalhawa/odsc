## Utility Functions

##  ultis -------------------------------------------------------------
show_space <- function(epoch=NULL,encoder,size,dataset,pre) {
  space  = data.table(V1=numeric(),V2=numeric(),class=factor())
  loops =  floor(size/batch_size)
  iter = make_iterator_one_shot(dataset)
  for (variable in 1:loops) { 
    x <-  iterator_get_next( iter)
    x_encoded = encoder( list(x,encoder_init_hidden)) %>% tfd_sample(1000) %>% k_mean(axis = 1) %>%  as.matrix() %>% as.data.table()
    x_encoded = x_encoded[, class := as.array(x$target) ]
    space = rbind(space,x_encoded)
  }
  p = ggplot(space, aes(x = V1, y = V2, colour = class))  + geom_point(position = 'jitter') 
  ggsave(filename = glue('plots/p{pre}{epoch}.png'),p)
  p
}
# show_space(epoch,size = 8000)

compute_space <- function(dat_points) {
  dat_iter <- tensor_slices_dataset(keras_array(dat_points,dtype = 'float32')) %>% 
    dataset_batch(batch_size,drop_remainder = T) %>%
    make_iterator_one_shot()
  scores = prob_table = matrix(,nrow = 0,ncol=3)
  colnames(scores)<- c('V1','V2','Score')
  until_out_of_range({  
    next_batch <- iterator_get_next(dat_iter) 
    prob_table  = as.matrix(next_batch) # 
    iter <-   make_iterator_one_shot(ch_dataset)
    pr = vector()
    until_out_of_range({
      x <-  iterator_get_next(iter)
      ampl = ifelse(x$target==0,-1,1)
      pr = cbind(pr, as.matrix(encoder_ts(list(x,  k_zeros(c(batch_size, n_gru), dtype='float32')  )) %>% tfd_prob(next_batch) * ampl ))
    })
    prob_table = cbind(prob_table,rowMeans(pr))
    colnames(prob_table) <- c('V1','V2','Score')
    scores = rbind(scores,prob_table)
  })
  round(scores,3)
}

compute_newpoint <- function(new_point,encoder) {
  new_points = encoder_ts(list(keras_array(new_point[,V1:V5]),  k_zeros(c(nrow(new_point), n_gru), dtype='float32')  )) %>% 
    tfd_sample(1000) %>% k_mean(axis=2)  %>% k_mean(axis = 1) %>% k_reshape(shape = c(1,2)) ## mean of 1000 samples from dist in latent space
  print(new_points)
  prob_table  = as.matrix(new_points) # 
  iter <-   make_iterator_one_shot(ch_dataset)
  pr = vector()
  until_out_of_range({
    x <-  iterator_get_next(iter)
    ampl = ifelse(x$target==0,-1,1)
    pr = cbind(pr, as.matrix(encoder(list(x,  k_zeros(c(batch_size, n_gru), dtype='float32')  )) %>% tfd_prob((new_points) ) * ampl ))
  })
  prob_table = cbind(prob_table,mean(pr),sd(pr))
  colnames(prob_table) <- c('V1','V2','Score','SD')
  # ALT..
  prob_table
 }

 nextBestAction <- function(sub_path,encoder) {
   channels = paste0('channel_', 0:8 )
   sub_path = cbind(sub_path[,V1:V4],V5=channels)
   out= lapply(seq(9) , function(s)  as.data.table(compute_newpoint(new_point = sub_path[s],encoder= encoder)) ) %>% rbindlist()
   sub_path = cbind(sub_path,out)
   colnames(sub_path) <-  c(paste0('step_', 1:5 ),names(out))
   sub_path
 }

